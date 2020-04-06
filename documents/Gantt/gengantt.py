import requests
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape

REPO_ADDR = 'https://api.github.com/repos/relientm96/capstone2020/'

class GithubInfo():
    def __init__(self):
        self.repo_addr = REPO_ADDR
        pass

    def get_mstone(self):
        milestones = requests.get(self.repo_addr + 'milestones')
        milestones_closed = requests.get(self.repo_addr + 'milestones?state=closed')
        mjson = milestones.json()
        mcljson = milestones_closed.json()
        self.mjson_all = mjson + mcljson
        for m in self.mjson_all:
            print('milestone:', m['title'])

    def get_issues(self):
        issues = requests.get(self.repo_addr + 'issues')
        issues_closed = requests.get(self.repo_addr + 'issues?q=is%3Aissue+is%3Aclosed')
        ijson = issues.json()
        icljson = issues_closed.json()
        self.ijson_all = ijson + icljson
        for i in self.ijson_all:
            print('issue:', i['title'])

    def get_json_from_github(self):
        self.get_mstone()
        self.get_issues()

class TexGen():
    def __init__(self):
        pass

    def gen_skeleton(self):
        # TODO: look into PyLaTeX
        self.doc = Document('gantt')
        with self.doc.create(Section('A section')):
            self.doc.append('hello world')
        self.doc.generate_pdf(clean_tex=False)

    def gen_gantt_chart(self):
        pass

if __name__ == "__main__":
    info = GithubInfo()
    info.get_json_from_github()
    texgen = TexGen()
    texgen.gen_skeleton()
