#!/usr/bin/env python3
"""仅用标准库顺序执行本目录的中文学习笔记，并保存可在 Jupyter 查看输出的副本。"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tokenize


def run(notebook_path, output_path=None):
    notebook_path = Path(notebook_path).resolve()
    document = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4 and document["nbformat_minor"] == 5
    cells = document["cells"]
    assert len({cell["id"] for cell in cells}) == len(cells), "单元标识必须唯一。"
    namespace = {"__name__": "__main__"}
    count = 0
    previous_directory = Path.cwd()
    try:
        # 与从仓库打开笔记时保持一致，源码定位不依赖调用者的目录。
        os.chdir(notebook_path.parents[2])
        for index, cell in enumerate(cells):
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            assert isinstance(cell["metadata"], dict)
            if cell["cell_type"] == "markdown":
                assert re.search(r"[\u4e00-\u9fff]", source), "说明单元必须包含中文。"
                continue
            assert cell["cell_type"] == "code", "本笔记只使用说明与代码单元。"
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    assert re.search(r"[\u4e00-\u9fff]", token.string), "新增代码注释应使用中文。"
            captured = io.StringIO()
            count += 1
            with contextlib.redirect_stdout(captured):
                # 无错误捕获后继续：任何单元失败都使验证立即失败，避免留下伪成功记录。
                exec(compile(source, f"{notebook_path.name}:单元{index + 1}", "exec"), namespace)
            cell["execution_count"] = count
            cell["outputs"] = []
            if captured.getvalue():
                cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": captured.getvalue()})
    finally:
        os.chdir(previous_directory)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"中文笔记执行通过：{count} 个代码单元，全部断言通过。")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行中文数学实验；不导入原生 CUDA 训练模块。")
    parser.add_argument("--notebook", type=Path, default=Path(__file__).with_name("gaussian_splatting_zh.ipynb"))
    parser.add_argument("--output", type=Path, help="可选：保存含本次执行结果的笔记；可指定原文件以更新输出。")
    options = parser.parse_args()
    run(options.notebook, options.output)
