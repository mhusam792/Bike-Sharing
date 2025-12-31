import pandas as pd
import pytest

from bike_sharing_model.data.loader import load_dataframe


@pytest.fixture
def sample_csv(tmp_path):
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    file_path = tmp_path / "test.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def empty_csv(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")  # ملف فارغ
    return file_path


@pytest.fixture
def non_csv_file(tmp_path):
    file_path = tmp_path / "not_csv.txt"
    file_path.write_text("this is not a csv file")
    return file_path


def test_load_valid_csv(sample_csv):
    df = load_dataframe(str(sample_csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df["col1"].tolist() == [1, 2, 3]
    assert df["col2"].tolist() == ["a", "b", "c"]


def test_load_empty_csv(empty_csv):
    with pytest.raises(ValueError) as excinfo:
        load_dataframe(str(empty_csv))
    assert "CSV file is empty" in str(excinfo.value)


def test_load_non_csv(non_csv_file):
    with pytest.raises(ValueError) as excinfo:
        load_dataframe(str(non_csv_file))
    assert "Failed to load CSV file" in str(excinfo.value)


def test_load_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(ValueError) as excinfo:
        load_dataframe(str(missing_file))
    assert "Failed to load CSV file" in str(excinfo.value)
