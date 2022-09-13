import pytest
from ccrec.util.demo_data import DemoData


@pytest.fixture
def demo_data_obj():
    return DemoData(mock_image=True)


def test_vae_main(demo_data_obj):
    return demo_data_obj.run_vae_main()


def test_bmt_main(demo_data_obj):
    return demo_data_obj.run_bmt_main()


def test_shap(demo_data_obj):
    return demo_data_obj.run_shap()
