"""
作业批改工具 - 基于 LangChain + Claude 多模态模型

使用方式:
    python homework_grader.py \
        --homework_dir ./作业文件夹 \
        --questions "1. 求解方程 x^2+2x+1=0\n2. 计算定积分..." \
        --output report.md

文件夹结构支持:
    方式1: 每人一个文件 (张三.jpg, 李四.pdf)
    方式2: 每人一个文件夹 (张三/page1.jpg, 张三/page2.jpg)
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
from collections import defaultdict
from pathlib import Path

import fitz  # pymupdf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# ───────────────── 配置 ─────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


# ───────────────── 文件扫描 ─────────────────


def scan_homework_dir(homework_dir: str) -> dict[str, list[Path]]:
    """扫描作业文件夹，按学生名分组文件。

    自动识别两种结构:
    - 每人一个文件: 文件名(不含后缀)即学生名
    - 每人一个文件夹: 文件夹名即学生名
    """
    root = Path(homework_dir)
    students: dict[str, list[Path]] = {}

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            # 方式2: 子文件夹 = 学生
            name = entry.name
            files = sorted(
                f
                for f in entry.iterdir()
                if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS
            )
            if files:
                students[name] = files
        elif entry.is_file() and entry.suffix.lower() in ALL_EXTENSIONS:
            # 方式1: 单文件 = 学生
            name = entry.stem
            students.setdefault(name, []).append(entry)

    return students


# ───────────────── 图片编码 ─────────────────


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[bytes]:
    """将 PDF 每页转为 PNG 图片字节。"""
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def load_student_images(files: list[Path]) -> list[str]:
    """将学生的所有作业文件转为 base64 编码图片列表。"""
    encoded = []
    for f in files:
        if f.suffix.lower() in PDF_EXTENSIONS:
            for img_bytes in pdf_to_images(f):
                encoded.append(encode_image(img_bytes))
        else:
            encoded.append(encode_image(f.read_bytes()))
    return encoded


# ───────────────── LangChain 批改链 ─────────────────

SYSTEM_PROMPT = """\
你是一位助教，负责汇总同学自批作业中的错题信息。

## 你的任务
同学已经自己批改了作业，把做错的题目写了出来，并给出了自己的分析和纠正。
请仔细查看提交的作业图片，提取以下信息:
1. 做错了哪些题（题号、题目内容）
2. 犯了什么错误
3. 自己给出的分析/纠正内容

注意：你不需要判断对错，只需要如实提取已标注的错题和分析。
在描述错误时，主语用"有同学"而不是"学生"。

## 输出要求
请严格按以下 JSON 格式返回结果，不要包含其他内容:
```json
{{
  "results": [
    {{
      "question_id": "题号，如 1, 2, 3",
      "question_summary": "题目的简要描述",
      "error_description": "犯了什么错误（用'有同学'作主语）",
      "student_analysis": "自己写的分析/纠正内容",
      "knowledge_point": "涉及的知识点"
    }}
  ],
  "error_count": 错题总数(整数),
  "overall_comment": "对错题情况的简要概括，1-2句话（用'有同学'而不是'学生'）"
}}
```
"""


def build_grading_message(
    questions: str, images_b64: list[str]
) -> list:
    """构建多模态消息，包含题目文本和学生作业图片。"""
    if questions.strip():
        intro = f"## 题目和要求\n{questions}\n\n## 学生作业\n以下是该学生提交的作业图片:"
    else:
        intro = "## 学生作业\n请从以下作业图片中自行识别题目和学生的作答内容，然后逐题评分:"
    content = [{"type": "text", "text": intro}]

    for i, img in enumerate(images_b64, 1):
        content.append({"type": "text", "text": f"第{i}张图片:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                },
            }
        )

    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]


def grade_student(
    llm: ChatOpenAI,
    parser: JsonOutputParser,
    questions: str,
    images_b64: list[str],
) -> dict:
    """调用模型批改单个学生的作业。"""
    messages = build_grading_message(questions, images_b64)
    response = llm.invoke(messages)
    try:
        return parser.invoke(response)
    except Exception:
        # 尝试从文本中提取 JSON
        text = response.content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"无法解析模型输出: {text[:200]}")


# ───────────────── 统计分析 ─────────────────


def group_questions_by_id(all_results: dict[str, dict]) -> dict[str, dict]:
    """按题号直接分组，不做相似题目合并。"""
    groups: dict[str, dict] = {}

    for name, r in all_results.items():
        for item in r.get("results", []):
            qid = str(item.get("question_id", ""))
            if qid not in groups:
                groups[qid] = {
                    "summary": item.get("question_summary", ""),
                    "knowledge_point": item.get("knowledge_point", ""),
                    "errors": [],
                }
            groups[qid]["errors"].append({
                "student": name,
                "error": item.get("error_description", ""),
                "analysis": item.get("student_analysis", ""),
            })
            if not groups[qid]["knowledge_point"] and item.get("knowledge_point"):
                groups[qid]["knowledge_point"] = item["knowledge_point"]

    return groups


def analyze_results(
    all_results: dict[str, dict],
    groups: dict[str, dict] = None,
) -> tuple[list[dict], dict[str, list[str]]]:
    """汇总所有错题，按题号分组统计出错人数。"""
    if groups is None:
        groups = {}

    ranked = []
    for qid, group in groups.items():
        unique_students = list(set(e["student"] for e in group["errors"]))
        ranked.append(
            {
                "question_id": qid,
                "question_summary": group["summary"],
                "knowledge_point": group["knowledge_point"],
                "count": len(unique_students),
                "students": unique_students,
                "errors": group["errors"],
            }
        )

    ranked.sort(key=lambda x: x["count"], reverse=True)
    return ranked[:5], {r["question_id"]: r["students"] for r in ranked}


def generate_error_analysis(
    llm: ChatOpenAI, questions: str, top_errors: list[dict]
) -> str:
    """对高频错题生成综合分析。"""
    if not top_errors:
        return "所有同学作业中没有标注错题。"

    error_info = ""
    for item in top_errors:
        error_info += (
            f"\n### 第{item['question_id']}题 — {item['question_summary']}\n"
            f"知识点: {item['knowledge_point']}\n"
            f"出错人数: {item['count']}人\n"
        )
        error_info += "各同学的错误和分析:\n"
        for e in item["errors"]:
            error_info += f"- {e['student']}: 错误={e['error']}; 分析={e['analysis']}\n"

    prompt = f"""\
以下是多位同学自批作业后汇总的错题信息:
{error_info}

请作为助教，对这些错题进行综合分析:
1. 每道题涉及什么知识点
2. 同学们普遍犯了什么类型的错误（归纳共性）
3. 出错的根本原因分析
4. 给任课老师的教学建议

请用中文回答，格式清晰，重点突出共性问题。
注意：在描述中用"有同学"而不是"学生"。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ───────────────── Markdown 报告 ─────────────────


def generate_report(
    all_results: dict[str, dict],
    top_errors: list[dict],
    error_analysis: str,
    questions: str,
    merged_groups: dict[str, dict] = None,
) -> str:
    """生成错题汇总 Markdown 报告。"""
    lines = ["# 错题汇总\n"]

    # 总览
    total_errors = sum(r.get("error_count", 0) for r in all_results.values())
    lines.append(f"{len(all_results)}个人，一共错���{total_errors}道题。\n")

    # 按合并后的题目分组，��出错人数排序
    sorted_questions = sorted(
        (merged_groups or {}).items(),
        key=lambda x: len(set(e["student"] for e in x[1]["errors"])),
        reverse=True,
    )

    lines.append("## 各题出错情况（Top 5）\n")
    for idx, (qid, group) in enumerate(sorted_questions[:5], 1):
        unique_students = len(set(e["student"] for e in group["errors"]))
        lines.append(f"### 第{idx}题 — {group['summary']}（{unique_students}人出错）\n")
        if group["knowledge_point"]:
            lines.append(f"知识点: {group['knowledge_point']}\n")
        for e in group["errors"]:
            lines.append(f"- {e['error']}")
            lines.append("")
        lines.append("---\n")

    # 高频错题排名
    lines.append("## 哪些题错的人最多\n")
    if top_errors:
        lines.append("| 排名 | 题目 | 错几个人 |")
        lines.append("|------|------|----------|")
        for i, item in enumerate(top_errors, 1):
            lines.append(
                f"| {i} | 第{item['question_id']}题：{item['question_summary']} "
                f"| {item['count']}人 |"
            )
        lines.append("")
    else:
        lines.append("没有发现错题。\n")

    # 综合分析
    lines.append("## 综合分析与教学建议\n")
    lines.append(error_analysis)
    lines.append("")

    return "\n".join(lines)


# ───────────────── 主流程 ─────────────────


def grade_homework(
    homework_dir: str,
    questions: str,
    output_path: str = "report.md",
    model: str = "qwen-vl-max",
    api_key=None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> str:
    """批改作业主函数。

    Args:
        homework_dir: 学生作业文件夹路径
        questions: 题目和要求文本
        output_path: 输出报告路径
        model: 模型名称
        api_key: DashScope API Key (默认读取环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY)
        base_url: 自定义 API 地址 (默认千问 DashScope 兼容模式)

    Returns:
        生成的报告内容
    """
    # 初始化模型
    llm_kwargs = {"model": model, "temperature": 0, "max_tokens": 4096}
    if api_key:
        llm_kwargs["openai_api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)
    parser = JsonOutputParser()

    # 扫描文件
    students = scan_homework_dir(homework_dir)
    if not students:
        raise FileNotFoundError(f"未在 {homework_dir} 中找到任何学生作业文件")

    print(f"找到 {len(students)} 位学生的作业:")
    for name, files in students.items():
        print(f"  - {name}: {len(files)} 个文件")
    print()

    # 并发提取错题
    all_results: dict[str, dict] = {}

    def _process_student(name_files):
        name, files = name_files
        images_b64 = load_student_images(files)
        result = grade_student(llm, parser, questions, images_b64)
        error_count = result.get("error_count", len(result.get("results", [])))
        return name, result, error_count

    max_workers = min(8, len(students))
    print(f"并发分析中 (最多 {max_workers} 个同时) ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_student, item): item[0]
            for item in students.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name, result, error_count = future.result()
            all_results[name] = result
            print(f"  {name} 完成 (发现 {error_count} 道错题)")

    # 按题号分组
    print("\n正在按题号分组 ...")
    groups = group_questions_by_id(all_results)

    # 统计分析
    print("正在统计分析 ...")
    top_errors, _ = analyze_results(all_results, groups)

    # 生成错题详细分析
    print("正在生成错题分析 ...")
    error_analysis = generate_error_analysis(llm, questions, top_errors)

    # 生成报告
    report = generate_report(all_results, top_errors, error_analysis, questions, groups)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\n报告已生成: {output_path}")

    return report


# ───────────────── CLI ─────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="作业批改工具 - 基于 LangChain + Claude")
    parser.add_argument("--homework_dir", required=True, help="学生作业文件夹路径")
    parser.add_argument("--questions", default="", help="题目和要求 (支持 \\n 换行，不提供则自动识别)")
    parser.add_argument("--output", default="report.md", help="输出报告路径 (默认 report.md)")
    parser.add_argument("--model", default="qwen-vl-max", help="模型名称 (默认 qwen-vl-max)")
    parser.add_argument("--api_key", default=None, help="DashScope API Key")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="自定义 API 地址 (默认千问 DashScope)")
    args = parser.parse_args()

    grade_homework(
        homework_dir=args.homework_dir,
        questions=args.questions.replace("\\n", "\n"),
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
