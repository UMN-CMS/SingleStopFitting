import jinja2
import os
from jinja2 import Template


def getEnv():
    latex_jinja_env = jinja2.Environment(
    block_start_string="\JB{",
    block_end_string="}",
    variable_start_string="\JV{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(os.path.abspath("./templates")),
)
    return latex_jinja_env

def renderTemplate(template_name, data):
    env = getEnv()
    template = env.get_template(template_name)
    return template.render(**data)
