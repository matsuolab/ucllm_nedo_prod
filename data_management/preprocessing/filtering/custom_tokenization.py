from hojichar import Filter, Document
from fugashi import Tagger

tagger = Tagger('-Owakati')


class WakatiTokenizer(Filter):
    """
    Tokenizer の実装例です.
    fugashi を用いて文を分割します.
    """

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> list[str]:
        """
        >>> WakatiTokenizer().tokenize("おはよう。おやすみ。ありがとう。さよなら。")
        ['おはよう。', 'おやすみ。', 'ありがとう。', 'さよなら。']
        """
        return tagger.parse(text).split()


class NewLineSentenceTokenizer(Filter):
    """
    日本語を想定したセンテンス単位のトーカナイザです.
    改行文字で文章を区切ります.
    """

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> list[str]:
        """
        >>> NewLineSentenceTokenizer().tokenize("おはよう。\nおやすみ。\nありがとう。\nさよなら。")
        ['おはよう。', 'おやすみ。', 'ありがとう。', 'さよなら。']
        """
        return text.split("\n")


class MergeTokens(Filter):
    """
    Mergerの実装例です.
    破棄されていないトークンをdelimeterで結合し, Document を更新します.
    """

    def __init__(self, delimiter: str = "", before_merge_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delimiter = delimiter

    def merge(self, tokens: list[str]) -> str:
        """
        >>> MergeTokens("\n").merge(["hoo", "bar"])
        'hoo\nbar'
        """
        return self.delimiter.join(tokens)

    def apply(self, document: Document) -> Document:
        remained_tokens = [token.text for token in document.tokens if not token.is_rejected]
        document.text = self.merge(remained_tokens)
        return document
