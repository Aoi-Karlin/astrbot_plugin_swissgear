import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register

# Covers the main Unicode emoji blocks; compiled once at import time.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F7FF"  # Alchemical Symbols
    "\U0001F780-\U0001F8FF"  # Geometric Shapes Extended
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FAFF"  # Chess Symbols, Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats (partial)
    "\U000024C2-\U000024FF"  # Enclosed Alphanumerics (safe range, e.g. ⓐ-⓿)
    "\U0001F170-\U0001F251"  # Enclosed Alphanumeric Supplement (🅰-🉑)
    "☀-⛿"  # Misc Symbols (U+2600-U+26FF)
    "✀-➿"  # Dingbats (U+2700-U+27BF)
    "︀-️"  # Variation Selectors
    "‍"    # Zero Width Joiner
    "⃣"    # Combining Enclosing Keycap
    "]+",
    flags=re.UNICODE,
)


@register(
    "astrbot_plugin_swissgear",
    "Abyss",
    "在 LLM 回复发送前清理多余换行与全/半角括号包裹的动作描述",
    "v1.1.1",
)
class SwissGearPlugin(Star):
    """瑞士军刀式回复清理插件。

    针对 DeepSeek 等指令遵循较弱的模型，在其回复真正发送前，
    自动去除括号内的动作/神态描述（如 （微笑着看向你）），
    并压缩多余的空行/换行。
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        # 当存在 _conf_schema.json 时，AstrBot 会注入配置对象（行为类似 dict）
        self.config = config or {}
        # 预编译括号正则，避免每条消息重复构建
        self._bracket_patterns = self._build_bracket_patterns(
            self.config.get("filter_brackets", ["（）", "()"])
        )

    def _build_bracket_patterns(self, pairs) -> list[re.Pattern]:
        """根据配置的括号对列表，构建对应的删除正则。

        每一项须为两个字符（左括号 + 右括号）。使用负向字符集实现
        非贪婪且跨行的匹配；非法项（长度不为 2、左右相同）会被跳过并记录。
        """
        patterns: list[re.Pattern] = []
        for item in pairs or []:
            s = str(item).strip()
            if len(s) != 2:
                logger.warning(f"[swissgear] 跳过非法括号配置项（须为两个字符）：{item!r}")
                continue
            left, right = s[0], s[1]
            if left == right:
                logger.warning(f"[swissgear] 跳过左右相同的括号配置项：{item!r}")
                continue
            le, re_ = re.escape(left), re.escape(right)
            # [^左右]* 保证非贪婪、不跨越同类括号，且天然支持跨行
            patterns.append(re.compile(f"{le}[^{le}{re_}]*{re_}"))
        return patterns

    def _clean_text(self, text: str) -> str:
        """清理单段文本：去除括号动作描述并压缩多余换行。

        清理是纯函数式的，不修改入参之外的任何状态。
        """
        if not text:
            return text

        # 1) 按配置逐类删除括号包裹的内容
        for pat in self._bracket_patterns:
            text = pat.sub("", text)

        # 2) 删除 Emoji
        if self.config.get("filter_emoji", False):
            text = _EMOJI_RE.sub("", text)

        # 3) 压缩多余空白与换行
        if self.config.get("collapse_blank_lines", True):
            max_nl = int(self.config.get("max_consecutive_newlines", 1) or 1)
            if max_nl < 1:
                max_nl = 1

            # 去除每行行尾的空白
            text = re.sub(r"[ \t]+\n", "\n", text)
            # 删除内容后残留的连续空格（例如 "你好 （笑） 走吧" -> "你好  走吧"）
            text = re.sub(r"[ \t]{2,}", " ", text)
            # 将超出上限的连续换行压缩到上限值
            pattern = r"\n{" + str(max_nl + 1) + r",}"
            text = re.sub(pattern, "\n" * max_nl, text)
            # 去除首尾空白
            text = text.strip()

        return text

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """在 LLM 响应被组装成消息前，清理其文本内容。"""
        try:
            if resp is None or not getattr(resp, "completion_text", None):
                return

            original = resp.completion_text
            cleaned = self._clean_text(original)

            if cleaned != original:
                resp.completion_text = cleaned
                reduction_rate = (len(original) - len(cleaned)) / len(original) if len(original) > 0 else 0

                if reduction_rate >= 0.9:
                    logger.warning(
                        f"[swissgear] 极端清理（减少 {reduction_rate*100:.1f}%）："
                        f"原长度={len(original)} -> 新长度={len(cleaned)}\n"
                        f"原文：{original!r}"
                    )
                else:
                    logger.info(
                        f"[swissgear] 已清理 LLM 回复："
                        f"原长度={len(original)} -> 新长度={len(cleaned)}"
                    )
        except Exception as e:
            # 单条消息处理失败不应影响整体流程
            logger.error(f"[swissgear] 清理 LLM 回复时出错：{e}")

    async def terminate(self):
        """插件卸载/停用时的清理逻辑。本插件无持久资源，留空即可。"""
        logger.info("[swissgear] 插件已停用。")
