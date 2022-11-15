import os, warnings, dataclasses, torch, time, socket, json, io, typing, shap
import pandas as pd, numpy as np, scipy.sparse as sps
from pprint import pprint
from rime.util import indices2csr, auto_tensor, timed, auto_device
from ccrec.env.base import Env, _expand_na_class
from ccrec.util.shap_explainer import I2IExplainer, plot_shap_values
import multiprocessing, functools
from urllib.request import urlopen

try:
    import boto3

    s3 = boto3.client("s3")
    sagemaker = boto3.client("sagemaker")
    iam = boto3.client("iam")
except ImportError:
    warnings.warn("Please install boto3 to use I2IEnv")
    s3 = sagemaker = iam = None

try:
    import matplotlib.pyplot as plt, tqdm
    from PIL import Image
    from textwrap import wrap
except ImportError:
    warnings.warn("Please install matplotlib, tqdm, PIL, textwrap to use I2IImageEnv")
    plt = tqdm = Image = wrap = None


def get_notebook_name():
    try:
        # https://stackoverflow.com/questions/59956042/how-to-return-name-of-aws-sagemaker-notebook-instance-within-python-script
        log_path = "/opt/ml/metadata/resource-metadata.json"
        with open(log_path, "r") as logs:
            _logs = json.load(logs)
        return _logs["ResourceName"]
    except Exception:
        return socket.gethostname().replace(".", "-")


def get_role_arn():
    if "CCREC_ROLE_ARN" in os.environ:
        return os.environ["CCREC_ROLE_ARN"]
    return [
        x["Arn"]
        for x in iam.list_roles(PathPrefix="/service-role/")["Roles"]
        if "AmazonSageMaker-ExecutionRole" in x["RoleName"]
    ][-1]


def get_s3_key(s3_path):
    parts = s3_path.split("/")
    return "/".join(parts[3:])


def text_ui_template(multi_label, prompt):
    tag = "crowd-classifier-multi-select" if multi_label else "crowd-classifier"
    return f"""
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form>
  <{tag}
    name="{tag}"
    categories="{{{{ task.input.labels | to_json | escape }}}}"
    header="{prompt}"
  >
    <classification-target style="white-space: pre-wrap">
      {{{{ task.input.taskObject }}}}
    </classification-target>
    <full-instructions header="Classifier instructions">
      <ol><li><strong>Read</strong> the text carefully.</li>
      <li><strong>Read</strong> the examples to understand more about the options.</li>
      <li><strong>Choose</strong> the appropriate labels that best suit the text.</li></ol>
    </full-instructions>
    <short-instructions>
      <p>{prompt}</p>
    </short-instructions>
  </{tag}>
  </crowd-form>
"""


@dataclasses.dataclass
class I2IConfig:
    s3_prefix: str = f"s3://{get_notebook_name()}-labeling/{get_notebook_name()}"
    role_arn: str = dataclasses.field(default_factory=lambda: get_role_arn())
    NumberOfHumanWorkersPerDataObject: int = 3
    PublicWorkforceTaskPrice: dict = dataclasses.field(
        default_factory=lambda: {
            "AmountInUsd": {"Dollars": 0, "Cents": 6, "TenthFractionsOfACent": 0}
        }
    )
    autorun: bool = True
    image: bool = False  # use in auto_select_env


@dataclasses.dataclass
class ExpInfo:
    bucket: str
    exp_name: str

    def __post_init__(self):
        output_suffix = f"{self.exp_name}/manifests/output/output.manifest"

        self.s3_input = f"s3://{self.bucket}/{self.exp_name}/input.manifest"
        self.s3_output = f"s3://{self.bucket}/{self.exp_name}"
        self.s3_output_manifest = f"s3://{self.bucket}/{self.exp_name}/{output_suffix}"
        self.s3_label_template = (
            f"s3://{self.bucket}/{self.exp_name}/support/label-template.json"
        )
        self.s3_ui_template = (
            f"s3://{self.bucket}/{self.exp_name}/support/template.liquid"
        )

    def __getitem__(self, key):
        return getattr(self, key)


@dataclasses.dataclass
class I2IEnv(Env):
    _is_synthetic: bool = False
    oracle: I2IConfig = "required"
    bucket: str = None
    multi_label: bool = False
    prompt: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.bucket is None:
            self.bucket = self.oracle.s3_prefix.split("/")[2]
        assert (
            self.test_requests["_hist_len"] > 0
        ).all(), "require item histories for i2i labeling"

        if self.prompt is None:
            if self.multi_label:
                self.prompt = "Please pick ALL items that are similar to the given item."
            else:
                self.prompt = "Please pick ONE item that is most similar to the given item."

    def _invoke(self, request, D, step_idx):
        exp_info = self._get_exp_info(step_idx)
        self._upload_request(request, exp_info)
        if self.oracle.autorun:
            self._run(exp_info)
        with timed("labeling job"):
            self._wait(exp_info)
        response = download_labels(exp_info, request)
        return response

    def _get_exp_info(self, step_idx=None, exp_name=None):
        if exp_name is None:
            exp_name = f"{get_s3_key(self.oracle.s3_prefix)}-{self.name}-ver-{self._logger.version}-step-{step_idx}"
        exp_info = ExpInfo(self.bucket, exp_name)
        pprint(exp_info)
        return exp_info

    def text_format(self, x):
        return "\n".join(
            [self.prompt]
            + [f'Given: "{x["last_title"]}"\n']
            + [f'{i+1}. "{c}"\n' for i, c in enumerate(x["cand_titles"])]
            + [f'{len(x["cand_titles"])+1}. (None of the above)']
        )

    def _upload_request(self, request, exp_info):
        num_valid_classes = len(request.iloc[0]["cand_titles"])
        request = request.assign(source=request.apply(self.text_format, axis=1))
        print(len(request), request["source"].iloc[0])

        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_input"]),
            Body=request.reset_index().to_json(orient="records", lines=True),
        )
        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_label_template"]),
            Body=json.dumps(
                {
                    "document-version": "2018-11-28",
                    "labels": [{"label": str(i + 1)} for i in range(num_valid_classes)]
                    + [{"label": "(None of the above)"}],
                }
            ),
        )
        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_ui_template"]),
            Body=text_ui_template(self.multi_label, self.prompt),
        )

    def _run(self, exp_info):
        print("Creating labeling job", exp_info["exp_name"])
        response = sagemaker.create_labeling_job(
            LabelingJobName=exp_info["exp_name"],
            LabelAttributeName=exp_info["exp_name"],
            InputConfig={
                "DataSource": {"S3DataSource": {"ManifestS3Uri": exp_info["s3_input"]}},
                "DataAttributes": {
                    "ContentClassifiers": [
                        "FreeOfPersonallyIdentifiableInformation",
                        "FreeOfAdultContent",
                    ]
                },
            },
            OutputConfig={"S3OutputPath": exp_info["s3_output"], "KmsKeyId": ""},
            RoleArn=self.oracle.role_arn,
            LabelCategoryConfigS3Uri=exp_info["s3_label_template"],
            StoppingConditions={"MaxPercentageOfInputDatasetLabeled": 100},
            HumanTaskConfig={
                "WorkteamArn": f"arn:aws:sagemaker:{s3.meta.region_name}:394669845002:workteam/public-crowd/default",
                "UiConfig": {"UiTemplateS3Uri": exp_info["s3_ui_template"]},
                "NumberOfHumanWorkersPerDataObject": self.oracle.NumberOfHumanWorkersPerDataObject,
                "TaskTimeLimitInSeconds": 300,
                "TaskAvailabilityLifetimeInSeconds": 21600,
                "MaxConcurrentTaskCount": 1000,
                "PublicWorkforceTaskPrice": self.oracle.PublicWorkforceTaskPrice,
                **self._human_task_config,
            },
        )
        print(response)

    @property
    def _human_task_config(self):
        suffix = "MultiClassMultiLabel" if self.multi_label else "MultiClass"
        title = "Multi Label" if self.multi_label else "Single Label"
        return {
            "PreHumanTaskLambdaArn": f"arn:aws:lambda:us-west-2:081040173940:function:PRE-Text{suffix}",
            "TaskKeywords": ["Text", "categorization", "classification"],
            "TaskTitle": f"Text Classification ({title}): item similarity",
            "TaskDescription": "Categorize text into specific classes",
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": f"arn:aws:lambda:us-west-2:081040173940:function:ACS-Text{suffix}",
            },
        }

    def _wait(self, exp_info):
        while True:
            try:
                s3.get_object(
                    Bucket=self.bucket, Key=get_s3_key(exp_info["s3_output_manifest"])
                )
                return
            except Exception:
                print(".", end="")
                time.sleep(5)


def download_labels(exp_info, request=None, verbose=True):
    try:
        obj = s3.get_object(
            Bucket=exp_info["bucket"], Key=get_s3_key(exp_info["s3_output_manifest"])
        )
    except Exception:
        print(exp_info)
        raise
    df_ground_truth = pd.read_json(obj["Body"], lines=True, convert_dates=False)
    if verbose:
        pprint(df_ground_truth.iloc[0].to_dict())
    if request is None:
        request = df_ground_truth.set_index(df_ground_truth.columns.tolist()[:2])

    label = df_ground_truth[exp_info["exp_name"]].values
    num_valid_classes = len(request.iloc[0]["cand_items"])
    multi_label = np.zeros((len(label), num_valid_classes + 1))
    for i, lab in enumerate(label):
        if isinstance(lab, list):
            for j in lab:
                multi_label[i, j] = 1 / len(lab)
        elif (
            lab < num_valid_classes + 1
        ):  # exclude nan labels due to GroundTruth failures
            multi_label[
                i, int(lab)
            ] = 1  # include "none of the above" labels as hard negatives

    return _expand_na_class(request).assign(multi_label=multi_label.tolist())


def _show_image(url_or_path, ax=plt):
    try:
        if url_or_path.startswith("http"):
            with urlopen(url_or_path) as downloaded:
                with Image.open(downloaded) as image:
                    ax.imshow(image)
        else:
            with Image.open(url_or_path) as image:
                ax.imshow(image)
    except Exception as e:
        warnings.warn(f"Image {url_or_path} not shown due to {e}")


@dataclasses.dataclass
class I2IImageEnv(I2IEnv):
    explainer: typing.Callable = None

    def __post_init__(self):
        super().__post_init__()
        assert "landingImage" in self.item_df, "require landingImage to generate images"

    def image_format(self, x):
        """
        from attrdict import AttrDict
        from PIL import Image
        Image.open(I2IImageEnv.image_format(
            self=AttrDict(item_df=pd.DataFrame(columns=['landingImage'], index=<item_id_list>),
                          explainer=<optional>),
            x={'_hist_items': [item_id], 'cand_items': [item_id, item_id]},
        )).show()
        """
        plt.ioff()
        ncols = len(x["cand_items"])
        fig = plt.figure(figsize=(12, 12))
        given = x["_hist_items"][-1]
        given_text = self.item_df.loc[given]["TITLE"]
        given_image = self.item_df.loc[given]["landingImage"]
        cand_texts = [
            self.item_df.loc[candidate]["TITLE"] for candidate in x["cand_items"]
        ]        
        cand_images = [
            self.item_df.loc[candidate]["landingImage"] for candidate in x["cand_items"]
        ]
        if self.summarizer is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            summary = []
            for text in cand_texts:
                batch = self.summarizer[0]([text], truncation=True, padding="longest", return_tensors="pt").to(device)
                translated = self.summarizer[1].generate(**batch)
                tgt_text = self.summarizer[0].batch_decode(translated, skip_special_tokens=True)
                summary.append(tgt_text[0])
            cand_texts = summary

        if hasattr(self, "explainer") and self.explainer is not None:
            if getattr(self, "color_source", True) is True:
                (given_text,), cand_texts = (
                    self.explainer([given_text]),
                    self.explainer([given_text], cand_texts),
                )
            else:
                cand_texts = self.explainer([given_text], cand_texts)

        ax = fig.add_subplot(6, 5, 1, frameon=False, xticks=[], yticks=[])
        ax.text(0.5, 0.9, "Question:", ha="center", va="center", fontsize=20, style='italic')

        # ax = fig.add_subplot(3, 5, 2, frameon=False, xticks=[], yticks=[])
        # _show_image(given_image, ax)
        # ax.text(0.5, 0.5, "3 levels of government", ha="center", va="center", fontsize=20)
 
        ax = fig.add_subplot(6, 5, (2, 5), frameon=False, xticks=[], yticks=[])
        if isinstance(given_text, shap._explanation.Explanation):
            plot_shap_values(
                0,
                0.9,
                given_text,
                width=80,
                nrows=5,
                fontsize=int(16 * min(4 / ncols, 1)),
            )
        else:
            ax.text(
                0,
                0.8,
                "\n".join(wrap(given_text, width=100)[:4]),
                fontsize=14,
                va="center",
            )

        ax = fig.add_subplot(6, 5, 6, frameon=False, xticks=[], yticks=[])
        prompt = "Please pick ONE item that is most similar to the given item."
        ax.text(
            0,
            1.3,
            prompt,
            fontsize=18,
            ha="left",
            va="center",
            style='italic',
        )

        for i, (image, text) in enumerate(zip(cand_images, cand_texts)):
            if isinstance(image, str):  # notnull
                ax = fig.add_subplot(
                    3, ncols, ncols + 1 + i, frameon=False, xticks=[], yticks=[]
                )
                _show_image(image, ax)
            ax = fig.add_subplot(
                6, 5, (i+1)*5+1, frameon=False, xticks=[], yticks=[]
            )
            ax.text(0., 0.75, "({})".format(i+1), ha="left", va="center", fontsize=16)
            ax.text(0., 1.2, "."*250, ha="left", va="center", fontsize=12)
            ax = fig.add_subplot(
                6, 5, ((i+1)*5+2, (i+1)*5+5), frameon=False, xticks=[], yticks=[]
            )
            if isinstance(text, shap._explanation.Explanation):
                plot_shap_values(
                    0,
                    0.9,
                    text,
                    width=80,
                    nrows=3,
                    fontsize=int(16 * min(4 / ncols, 1)),
                )
            else:
                ax.text(
                    0,
                    0.5,
                    "\n".join(wrap(f"{i+1}. " + text, width=20)[:6]),
                    fontsize=int(14 * min(4 / ncols, 1)),
                    va="center",
                )
        ax = fig.add_subplot(
                6, 5, 26, frameon=False, xticks=[], yticks=[]
            )
        ax.text(0., 0.75, "(5)", ha="left", va="center", fontsize=16)
        ax.text(0., 1.2, "."*250, ha="left", va="center", fontsize=12)
        ax = fig.add_subplot(
                6, 5, (27, 30), frameon=False, xticks=[], yticks=[]
            )
        ax.text(0., 0.75, "None of the above", fontsize=16, va="center")
        img_data = io.BytesIO()
        fig.savefig(img_data, format="jpg", transparent=False, bbox_inches="tight")
        img_data.seek(0)
        plt.close(fig)
        plt.ion()
        return img_data

    def _upload_image(self, i_img_data, exp_info):
        i, img_data = i_img_data
        img_data.seek(0)
        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_output"]) + f"/images/{i}.jpg",
            Body=img_data,
            ContentType="image/jpeg",
        )

    def _upload_request(self, request, exp_info):
        num_valid_classes = len(request.iloc[0]["cand_titles"])
        if self.explainer is None:
            with multiprocessing.Pool() as pool:
                images = list(
                    pool.map(
                        self.image_format, tqdm.tqdm(request.to_dict(orient="records"))
                    )
                )
        else:
            images = list(map(self.image_format, tqdm.tqdm(request.to_dict(orient="records"))))
        Image.open(images[0]).show()

        if self.explainer is None:
            with multiprocessing.Pool() as pool:
                pool.map(
                    functools.partial(self._upload_image, exp_info=exp_info),
                    enumerate(images),
                )
        else:
            list(map(functools.partial(self._upload_image, exp_info=exp_info), enumerate(images)))
        request = request.assign(
            **{
                "source-ref": [
                    exp_info["s3_output"] + f"/images/{i}.jpg"
                    for i in range(len(images))
                ]
            }
        )
        print(len(request), request.iloc[0])

        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_input"]),
            Body=request.reset_index().to_json(orient="records", lines=True),
        )
        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_label_template"]),
            Body=json.dumps(
                {
                    "document-version": "2018-11-28",
                    "labels": [{"label": str(i + 1)} for i in range(num_valid_classes)]
                    + [{"label": "(None of the above)"}],
                }
            ),
        )
        s3.put_object(
            Bucket=self.bucket,
            Key=get_s3_key(exp_info["s3_ui_template"]),
            Body=image_ui_template(self.multi_label, self.prompt),
        )

    @property
    def _human_task_config(self):
        suffix = "MultiClassMultiLabel" if self.multi_label else "MultiClass"
        return {
            "PreHumanTaskLambdaArn": f"arn:aws:lambda:us-west-2:081040173940:function:PRE-Image{suffix}",
            "TaskKeywords": ["Image classification"],
            "TaskTitle": "Image classification task",
            "TaskDescription": "Preference learning through images",
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": f"arn:aws:lambda:us-west-2:081040173940:function:ACS-Image{suffix}"
            },
        }


def image_ui_template(multi_label, prompt):
    tag = (
        "crowd-image-classifier-multi-select"
        if multi_label
        else "crowd-image-classifier"
    )
    return f"""
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form>
  <{tag}
    name="{tag}"
    src="{{{{ task.input.taskObject | grant_read_access }}}}"
    header="{prompt}"
    categories="{{{{ task.input.labels | to_json | escape }}}}"
  >
    <full-instructions header="Image classification instructions">
      <ol><li><strong>Read</strong> the task carefully and inspect the image.</li>
      <li><strong>Read</strong> the options and review the examples provided to understand more about the labels.</li>
      <li><strong>Choose</strong> the appropriate label that best suits the image.</li></ol>
    </full-instructions>
    <short-instructions>
      <p>{prompt}</p>
    </short-instructions>
  </{tag}>
</crowd-form>
"""
