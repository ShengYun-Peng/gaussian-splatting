#!/usr/bin/env python3
"""验证中文学习版保留原 Python 执行代码、子模块版本及 CUDA 源码原始行。"""

import ast
import io
from pathlib import Path
import subprocess
import tokenize


ROOT = Path(__file__).resolve().parents[2]
BASE = "54c035f7834b564019656c3e3fcc3646292f727d"
RASTERIZER = "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0"


def git(*arguments, cwd=ROOT):
    return subprocess.check_output(["git", *arguments], cwd=cwd).decode("utf-8")


def tokens(source):
    # 注释与非语义换行可变；名称、字符串、缩进及所有执行符号必须完全一致。
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    return [(token.type, token.string) for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in ignored]


def inserted_comments_only(original, annotated, prefix):
    # 保留每一行原文且顺序不变；只有整行注释允许插入，版权和既有说明同样必须保留。
    original_lines = original.splitlines(keepends=True)
    cursor = 0
    added = 0
    for line in annotated.splitlines(keepends=True):
        if cursor < len(original_lines) and line == original_lines[cursor]:
            cursor += 1
        else:
            assert line.lstrip().startswith(prefix), f"发现非注释插入或原行变动：{line!r}"
            content = line.rstrip("\r\n")
            assert content == content.rstrip(" \t"), f"新增注释有行尾空白：{line!r}"
            added += 1
    assert cursor == len(original_lines), "原始行被删除或移动。"
    return added


def check_python(original, annotated, path):
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(annotated)), f"语法树改变：{path}"
    assert tokens(original) == tokens(annotated), f"执行符号改变：{path}"
    compile(annotated, str(path), "exec")
    return inserted_comments_only(original, annotated, "#")


def main():
    python_count = changed_count = comment_count = 0
    for relative in git("ls-tree", "-r", "--name-only", BASE).splitlines():
        if not relative.endswith(".py"):
            continue
        original = git("show", f"{BASE}:{relative}")
        annotated = (ROOT / relative).read_text(encoding="utf-8")
        added = check_python(original, annotated, relative)
        python_count += 1
        changed_count += bool(added)
        comment_count += added
    print(f"原有 Python 文件：{python_count} 个全部通过语法树、执行符号、原始行和语法检查。")
    print(f"其中 {changed_count} 个文件新增中文注释 {comment_count} 行。")

    for record in git("ls-tree", "-r", BASE).splitlines():
        metadata, relative = record.split("\t", 1)
        mode, kind, revision = metadata.split()
        if mode == "160000":
            actual = git("rev-parse", "HEAD", cwd=ROOT / relative).strip()
            assert actual == revision, f"子模块版本改变：{relative}"
    print("所有子模块仍锁定原版本。")

    submodule = ROOT / "submodules/diff-gaussian-rasterization"
    mirror_root = Path(__file__).with_name("cuda-source")
    mirror_count = mirror_comments = 0
    for path in sorted(mirror_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".cu", ".h", ".cpp"}:
            continue
        relative = path.relative_to(mirror_root)
        original = git("show", f"{RASTERIZER}:{relative.as_posix()}", cwd=submodule)
        annotated = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            added = check_python(original, annotated, path)
        else:
            # 此处保证原始 C++ 行逐字保留且仅插入 // 整行注释；另外的词法检查记录见验证文档。
            added = inserted_comments_only(original, annotated, "//")
        mirror_count += 1
        mirror_comments += added
    assert mirror_count > 0, "缺少 CUDA 中文源码副本。"
    patch = Path(__file__).with_name("rasterizer-zh.patch")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=submodule, check=True)
    print(f"光栅器学习源码：{mirror_count} 个文件，仅插入注释 {mirror_comments} 行；补丁可干净应用。")
    print("这些检查证明注释没有改动原执行源码；不代替 CUDA 编译或数据集训练测试。")


if __name__ == "__main__":
    main()
