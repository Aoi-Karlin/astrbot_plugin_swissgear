import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register

# Covers the main Unicode emoji blocks; compiled once at import time.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F7FF"
    "\U0001F780-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "☀-⛿"
    "✀-➿"
    "︀-️"
    "‍"
    "⃣"
    "]+",
    flags=re.UNICODE,
)

# 匹配 Markdown 链接/引用，如 [标题](https://xxx) 或 [](https://xxx)。
# 用于在括号过滤前把整个链接保护起来，避免 () 过滤器只吃掉链接里的
# "(网址)" 部分，导致残留一个孤立的 "[]"（或 "[标题]"）。
_MD_LINK_RE = re.compile(r"\[[^\[\]]*\]\([^()]*\)")

# 保护占位符使用私用区（Private Use Area）字符包裹，正常文本几乎不可能
# 出现，能安全地在清理流程中原样穿过而不被空白压缩/括号过滤等步骤破坏。
_PLACEHOLDER_RE = re.compile("\uE000(\\d+)\uE001")


def _preview(text: str, limit: int = 80) -> str:
    """把文本压成单行、截断到指定长度，方便塞进一行日志里查看。"""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) > limit:
        flat = flat[:limit] + "…"
    return flat


@register(
    "astrbot_plugin_swissgear",
    "Abyss",
    "在 LLM 回复发送前清理多余换行与全/半角括号包裹的动作描述",
    "v1.1.2",
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

    @staticmethod
    def _strip_pair(pattern: re.Pattern, text: str) -> str:
        """反复应用括号删除正则直至不再变化。

        单次 sub() 无法处理同类括号嵌套的情况（如 "（外层（内层）文字）"）：
        由于中间的字符类禁止再次出现同类括号，第一次只能匹配到最内层，
        外层括号会被留下来。这里循环删除，内层删完后外层自然暴露出来，
        再次匹配删除，直到文本不再变化为止（设置上限防止极端输入卡死）。
        """
        for _ in range(50):
            new_text = pattern.sub("", text)
            if new_text == text:
                break
            text = new_text
        return text

    def _clean_text(self, text: str) -> str:
        """清理单段文本：去除括号动作描述并压缩多余换行。

        清理是纯函数式的，不修改入参之外的任何状态。
        """
        if not text:
            return text

        # 0) 保护 Markdown 链接（如 [标题](url)、[](url)），防止后续括号
        #    过滤把其中的圆括号部分单独吃掉，残留孤立的方括号
        protected: list[str] = []

        def _protect(m: re.Match) -> str:
            protected.append(m.group(0))
            return f"\uE000{len(protected) - 1}\uE001"

        text = _MD_LINK_RE.sub(_protect, text)

        # 1) 按配置逐类删除括号包裹的内容（循环处理同类括号嵌套的情况）
        for pat in self._bracket_patterns:
            text = self._strip_pair(pat, text)

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

        # 4) 还原之前被保护的 Markdown 链接
        if protected:
            text = _PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], text)

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
                # INFO 级：默认日志级别下也能看到，方便定位"回复被清空/异常删减"等问题
                logger.info(
                    "[swissgear] 已清理 LLM 回复："
                    f"原长度={len(original)} -> 新长度={len(cleaned)}"
                    f" | 清理前='{_preview(original)}'"
                    f" | 清理后='{_preview(cleaned)}'"
                )
                if not cleaned.strip():
                    # 清理后内容变为空，大概率是过滤规则误伤，单独告警便于排查
                    logger.warning(
                        "[swissgear] 清理后内容为空，原始回复可能被过度清理："
                        f"'{_preview(original, 200)}'"
                    )
        except Exception as e:
            # 单条消息处理失败不应影响整体流程
            logger.error(f"[swissgear] 清理 LLM 回复时出错：{e}")

    async def terminate(self):
        """插件卸载/停用时的清理逻辑。本插件无持久资源，留空即可。"""
        logger.info("[swissgear] 插件已停用。")
