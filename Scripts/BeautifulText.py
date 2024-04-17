from IPython.display import Markdown, display

class BeautifulText():
    """
    Display HTML format Markdown text in the notebook. Useful for highlighting important notes in
    the notebook.
    """
    def __init__(self, **kwargs):
        """
        Initialise with HTML style elements.
        Parameters
        ----------
        **kwargs: dict
            HTML style attributes for formatting the text.
        """
        self.style_attributes = kwargs

    def print_beautiful(self, text: str) -> None:
        """Applies the html attributes to `text` and prints it as MarkDown."""
        style_string = self._generate_style_string()
        beautifultext = f"""<span style='{style_string}'>{text}</span>"""
        display(Markdown(beautifultext))

    def _generate_style_string(self) -> str:
        """Generate the style string from the provided style attributes."""
        style_string = ""
        for attribute, value in self.style_attributes.items():
            style_string += f"{attribute}: {value}; "
        return style_string.strip()


# Example usage
comic = BeautifulText(font_family='Comic Sans MS', color='green', font_size=20)
comic.print_beautiful("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque cursus eros eget mi commodo, in ultricies lacus vulputate. Ut vel dui a augue feugiat mollis. Phasellus malesuada diam dui, eget accumsan nunc suscipit et. Cras et vestibulum sapien. Quisque a dolor vel nisl tempor cursus. Praesent sed sagittis eros. Mauris sed ultricies metus.")
