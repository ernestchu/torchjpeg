# pylint: disable=missing-function-docstring
from typing import List

from colorama import Fore, Style
from invoke import Exit, task


@task
def lint(c, which="all"):
    lint_on: List[str] = []
    if which == "all":
        lint_on += ["torchjpeg", "test", "examples", "./*.py"]
    else:
        lint_on.append(which)

    print(f"{Fore.BLUE}Linting: {Fore.GREEN}{lint_on}{Style.RESET_ALL}")

    for t in lint_on:
        r = c.run(f"pylint {t}", warn=True)
        if r.exited != 0:
            print(f"{Fore.RED}Linting for {Fore.GREEN}{t}{Fore.RED} FAILED{Style.RESET_ALL}")


@task
def type_checking(c, which="all"):
    tc_on: List[str] = []
    if which == "all":
        tc_on += ["src/torchjpeg", "test", "examples", "./*.py"]
    else:
        tc_on.append(which)

    print(f"{Fore.BLUE}Type Checking: {Fore.GREEN}{tc_on}{Style.RESET_ALL}")

    for t in tc_on:
        r = c.run(f"mypy {t}", warn=True)
        if r.exited != 0:
            print(f"{Fore.RED}Type checking FAILED{Style.RESET_ALL}")


@task
def sort_imports(c):
    print(f"{Fore.BLUE}Sorting Imports{Style.RESET_ALL}")
    c.run("isort . --recursive", warn=True)


@task
def style(c):
    print(f"{Fore.BLUE}Styling Code{Style.RESET_ALL}")
    c.run("black .", warn=True)


@task
def dco(c):
    print(f"{Fore.BLUE}Checking For DCO{Style.RESET_ALL}")

    c.run("git remote add upstream-dco-check https://gitlab.com/Queuecumber/torchjpeg.git", warn=True)
    c.run("git fetch -n upstream-dco-check", warn=True)
    r = c.run('git rev-list HEAD ^remotes/upstream-dco-check/master --count --grep "Signed-off-by: .*( <.*@.*>)?$" --invert-grep --extended-regexp', warn=True)
    c.run("git remote remove upstream-dco-check", warn=True)

    cnt = int(r.stdout)

    if cnt > 0:
        print(f"{Fore.RED}You have {Fore.GREEN}{cnt}{Fore.RED} commits missing DCO{Style.RESET_ALL}")
        print(f"{Fore.RED}DCO check FAILED{Style.RESET_ALL}")

        raise Exit(code=1)


@task
def cq(c):
    print(f"{Fore.YELLOW}Code Quality{Style.RESET_ALL}")

    try:
        dco(c)
    finally:
        sort_imports(c)
        style(c)
        type_checking(c)
        lint(c)
