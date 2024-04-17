from IPython.display import Markdown, display

class BeautifulText():
    """
    Display HTML format Markdown text in the notebook. Useful for highlighting important notes in
    the notebook.
    """
    def __init__(self, color: str = 'black', font_family: str = 'serif',
                 font_weight: str = 'normal', font_size: int = 16, font_style: str = 'normal',
                 background_color: str = 'none', letter_spacing: int = 1, line_height: float = 0.8,
                 word_spacing: int = 1, text_decoration: str = 'none', text_shadow: str = 'none'):
        """
        Initialise with HTML style elements.
        Parameters
        ----------
        color : str
            Color of the markdown text. Defaults to `black`.
        font_family : str
            `font-family` html attribute. Defaults to `serif`.
        font_weight : str
            `font-weight` html attribute. Defaults to `normal`.
        font_size : int
            `font-size` html attribute. Defaults to `16`px. (Don't include `px` in the argument.)
        font_style : str
            `font-style` html attribute. Defaults to `normal`.
        background_color : str
            `background-color` html attribute. Defaults to `none`
        letter_spacing : int
            `letter-spacing` html attribute. Defaults to `1`px.
            (Don't include `px` in the argument.)
        line_height : float
            `line-height` html attribute. Defaults to `0.8`px. (Don't include `px` in the argument.)
        word_spacing : int
            `word-spacing` html attribute. Defaults to `1`px. (Don't include `px` in the argument.)
        text-decoration : str
            `text-decoration` html attribute. Defaults to `none`.
        text-shadow : str
            `text-shadow` html attribute. Defaults to `none`.
        """
        self.color = color
        self.font_family = font_family
        self.font_weight = font_weight
        self.font_size = font_size
        self.font_style = font_style
        self.background_color = background_color
        self.letter_spacing = letter_spacing
        self.line_height = line_height
        self.word_spacing = word_spacing
        self.text_decoration = text_decoration
        self.text_shadow = text_shadow
    def printbeautiful(self, text: str) -> None:
        """Applies the html attributes to `text` and prints it as MarkDown."""
        beautifultext = f"""<span style='color: {self.color};
                        font-family: {self.font_family};
                        font-weight: {self.font_weight};
                        font-size: {self.font_size}px;
                        font-style: {self.font_style};
                        text-decoration: {self.text_decoration};
                        letter-spacing: {self.letter_spacing}px;
                        line-height: {self.line_height}px;
                        word-spacing: {self.word_spacing}px;
                        text-shadow: {self.text_shadow};
                        background-color: {self.background_color};'>
                        {text}
                        </span>
                        """
        display(Markdown(beautifultext))
