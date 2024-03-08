from hojichar import TokenFilter, Token
from fugashi import Tagger
import re

tagger = Tagger('-Owakati')


class RemoveDate(TokenFilter):
    """
    TokenFilter の実装例です.
    日付のみのパターンを削除
    """

    def __init__(self, date_pattern: re.Pattern = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.date_pattern = self._date_pattern() if date_pattern is None else date_pattern

    def _date_pattern(self) -> str:
        return re.compile(r'^(\d{2,4}([-年/])\d{1,2}([-月/])\d{1,2}日?)|(\d{2,4}([-年/])\d{1,2}([-月])?)|(\d{1,2}([-月/])\d{1,2}日?)$')

    def apply(self, token: Token) -> Token:
        text = token.text
        if self.date_pattern.match(text):
            token.is_rejected = True

        return token


class RemoveOneword(TokenFilter):
    """
    TokenFilter の実装例です.
    1単語のみのパターンを削除
    """

    def apply(self, token: Token) -> Token:
        text = token.text
        word_count = len(tagger.parse(text).split())
        if word_count <= 1:
            token.is_rejected = True

        return token
