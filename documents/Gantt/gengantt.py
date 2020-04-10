import requests
import subprocess
import os
import datetime
import calendar
import pandas as pd
# from pylatex import Document, Section, Subsection, Command
# from pylatex.utils import italic, NoEscape

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
        self.mjson = mjson + mcljson
        print('Retrieved milestones.')
        # for m in self.mjson:
        #     print('milestone:', m['title'])

    def get_issues(self):
        issues = requests.get(self.repo_addr + 'issues')
        issues_closed = requests.get(self.repo_addr + 'issues?q=is%3Aissue+is%3Aclosed')
        ijson = issues.json()
        icljson = issues_closed.json()
        self.ijson = ijson + icljson
        print('Retrieved issues.')
        # for i in self.ijson:
        #     print('issue:', i['title'])

    def get_proj(self):
        headers={'Accept':'application/vnd.github.inertia-preview+json',\
            'Authorization': 'token ' + self.token}
        projects = requests.get(self.repo_addr + 'projects', headers=headers)
        projects_closed = requests.get(self.repo_addr + 'projects?query=is%3Aclosed',\
            headers=headers)
        pjson = projects.json()
        pcljson = projects_closed.json()
        self.pjson = pjson + pcljson
        print('Retrieved projects.')

    def get_auth(self):
        self.user = 'tszkiup'
        self.token = '96fa62363797672f5141bd789d325ce9495907a6'
        #self.user = input('Please enter username: ')
        #self.password = getpass(prompt='Please enter your password: ')
        #self.token = getpass(prompt='Please enter personal token: ')

    def get_json_from_github(self):
        self.get_auth()
        self.get_mstone()
        self.get_issues()
        self.get_proj()

class ExtraInfo():
    def __init__(self):
        self.cap_start_date = datetime.date(2020, 3, 2)
        self.cap_end_date = datetime.date(2020, 11, 27)
        self.cap_dur = self.cap_end_date - self.cap_start_date

class TexGen():
    def __init__(self, info):
        self.filename = 'gantt.tex'
        self.texfile = open(self.filename, 'w')
        self.indent = 0
        self.cap_start_date = datetime.date(2020, 3, 2)
        self.cap_end_date = datetime.date(2020, 11, 27)
        self.info = info
        self.einfo = ExtraInfo()

    def gen_gantt(self):
        '''
        method for generating the whole document
        '''
        self.gen_config()
        self.begin_doc()
        self.gen_gantt_chart()
        self.end_doc()
        self.texfile.close()
        self.gen_pdf()

    def gen_config(self):
        self.print2tex('\\documentclass{article}')
        self.print2tex('\\usepackage{pgfgantt}')
        self.print2tex('\\usepackage[graphics,tightpage,active]{preview}')
        self.print2tex('\\PreviewEnvironment{tikzpicture}')
        self.print2tex('\\newlength{\imagewidth}')
        self.print2tex('\\newlength{\imagescale}')
        self.print2tex('\\pagestyle{empty}')
        self.print2tex('\\thispagestyle{empty}')

    def begin_doc(self):
        self.print2tex('\\begin{document}')

    def end_doc(self):
        self.print2tex('\\end{document}')

    def gen_gantt_chart(self):
        '''
        method for generating the gantt chart
        '''
        self.print_line('\\begin{ganttchart}')
        self.gantt_load_config()
        self.gantt_gen_title()
        self.gantt_gen_tasks()
        self.print_line('\\end{ganttchart}')

    def gantt_load_config(self):
        self.print_line('[')
        self.add_indent()
        self.print_line('x unit=0.1cm,')
        self.print_line('bar/.style={fill=gray!50},')
        self.print_line('incomplete/.style={fill=white},')
        self.rm_indent()
        self.print_line(']{1}{' + str(self.einfo.cap_dur.days+1) + '}')

    def gantt_gen_title(self):
        pf = '\\gantttitle'
        sdate = self.cap_start_date
        edate = self.cap_end_date
        self.print_line(pf+'{Month}{' + str(self.einfo.cap_dur.days+1) + '} \\\\')
        #for month in pd.date_range(self.einfo.cap_start_date, self.einfo.cap_end_date, freq='M'):
        self.print_line(pf+'{'+calendar.month_name[sdate.month]+'}{'\
            +str(calendar.monthrange(2020, sdate.month)[1]-sdate.day+1)+'}')
        for m in range(sdate.month+1, edate.month):
            self.print_line(pf+'{'+calendar.month_name[m]+'}{'\
                +str(calendar.monthrange(2020, m)[1])+'}')
        self.print_line(pf+'{'+calendar.month_name[edate.month]+'}{'\
            +str(edate.day)+'}')

    def gantt_gen_tasks(self):
        pf = '\\ganttbar'
        sdate = self.cap_start_date
        edate = self.cap_end_date

    def print_line(self, string):
        self.print2tex(' '*self.indent + string)

    def print2tex(self, string, end='\n'):
        self.texfile.write(string + end)

    def add_indent(self):
        self.indent += 4

    def rm_indent(self):
        self.indent -= 4

    def gen_pdf(self):
        # subprocess.run(['pdflatex', self.filename])
        # os.remove(os.path.splitext(self.filename)[0] + '.log')
        # os.remove(os.path.splitext(self.filename)[0] + '.aux')
        # print('Removed .log and .aux files.')
        print('Finished.')

if __name__ == "__main__":
    info = GithubInfo()
    info.get_json_from_github()
    texgen = TexGen(info)
    texgen.gen_gantt()
