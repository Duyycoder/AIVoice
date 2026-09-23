# -*- coding: utf-8 -*-
"""So thu tu cua video trong mot lo.

Ghep sai thu tu la loi nguoi dung chi phat hien sau khi ngoi xem het video da
ghep, nen phan danh so o day duoc kiem ky hon phan con lai cua adapter.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_MC_ROOT = Path(__file__).resolve().parents[1]
if str(_MC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MC_ROOT))


def _load_adapter():
    path = _MC_ROOT / "adapter_download_cli.py"
    spec = importlib.util.spec_from_file_location("adapter_download_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def _args(urls, batch_index=None, urls_file=""):
    return argparse.Namespace(url=list(urls), batch_index=list(batch_index or []),
                              urls_file=urls_file)


def test_moi_link_giu_dung_so_thu_tu_duoc_giao():
    args = _args(["https://a/1", "https://a/2", "https://a/3"], [5.0, 2.0, 9.0])
    assert adapter.read_urls(args) == [
        ("https://a/1", 5.0), ("https://a/2", 2.0), ("https://a/3", 9.0)]


def test_bo_link_trung_thi_bo_kem_so_thu_tu_cua_no():
    # Bo trung ma khong bo kem so cua no thi moi link phia sau deu lech mot cho.
    args = _args(["https://a/1", "https://a/1", "https://a/2"], [1.0, 2.0, 3.0])
    assert adapter.read_urls(args) == [("https://a/1", 1.0), ("https://a/2", 3.0)]


def test_khong_truyen_so_thu_tu_thi_dem_theo_vi_tri():
    args = _args(["https://a/1", "https://a/2"])
    assert adapter.read_urls(args) == [("https://a/1", 1.0), ("https://a/2", 2.0)]


def test_playlist_nam_gon_giua_hai_muc_dung_canh():
    # Link thu 2 cua lo giai ra 3 video: ca ba phai nam GIUA muc 2 va muc 3,
    # khong duoc day cac muc dung sau trong lo lech di.
    idxs = adapter.batch_indexes(2.0, 3)
    assert idxs == pytest.approx([2.001, 2.002, 2.003])
    assert all(2.0 < i < 3.0 for i in idxs)


def test_link_le_giu_nguyen_so_cua_minh():
    assert adapter.batch_indexes(7.0, 1) == [7.0]
    assert adapter.batch_indexes(7.0, 0) == [7.0]
