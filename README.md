# mthm503

![CI](https://github.com/<YOUR USER NAME>/<YOUR REPOSITORY NAME>/actions/workflows/main.yml/badge.svg)

This is a template for course MTHM503 "Applications of Data Science and Statistics".  You should fork this repository and use it for your work on this module

Note: in a professional setting, a DevOps engineer (or team of engineers) would be responsible for setting up infrastructure like this. You only have to operate within the infrastructure. 

## Setting up for the first time

I have tested the set up on a University Virtual Machine, a University Laptop, as well as my own Debian machines.  I have checked it on University Lab Machines. It does work, but occasionally Windows can make things difficult.  You really do need to follow all the instructions carefully.  The most important of these is: **Execute the run file in the Anaconda prompt**.

- Make sure you first have a GitHub account, and secondly, register this as a student account with GitHub.  More detailed information is given in ELE.
- Open Rstudio and create a new project under version control, using the GitHub repository you have just set up. There are slightly different instructions if you are doing this on a virtual machine.
- This is the tricky part. You need to move over to Anaconda Prompt and point the prompt at the repo you have just downloaded.   `cd` changes the directory, but more specific advice is not available as it all depends where you have put the local copy of your repository
- With Anaconda prompt in your repository folder you can execute `conda env create -f environment.yml`.  This will set up a conda environment on your working PC and will take a few moments. You then need to activate it by issuing the command  `activate python-exercises`
- Now, you have to restart your Rstudio.  If you can find the `python-exercises` conda environment all is good. If not, you may have to run the `start_repl.R` script which seems to magically set up some paths inside Rstudio. If you need to do this, open the `start_repl.R` script in Rstudio and hit the button for `source script`.
- The most important thing is, you need to do this in the first week of the course. Not much else will run if we don't get this set up right. You won't be able to submit coursework.

## Developing reproducible pipelines

The basic method of working is to write functions that pass unit tests.  As you progress, and especially if you want a higher mark, you will want to write your own unit tests and extend the pipelines.  We will look at that in due course.  In the meantime, do note:

1. The `REPL` (you can launch this anytime you want by sourcing the `start_repl.R` script) is an interactive python "shell". You can type commands in there e.g. `2 + 2` and view the result.  
2. You can also write pieces of code in a script file and `run` them in Python by (a) highlighting a region and clicking Ctrl + Enter () or (b) just pressing Ctrl + Enter to submit the cursor line.  If you don't have a python REPL open at this point, Rstudio will open one for you.
3. You can therefore "experiment" with pieces of code, see how they work line by line and so on. When you are ready, you can wrap these into a function.
4. The `doit` pipeline tool writes all intermediate stages out as files. You can therefore read in any of these as you see fit and work with it interactively, until you are ready to write a function.



## Formative feedback

To submit your work for formative feedback, you'll use a Pull Request (PR) on GitHub. This allows you to share your code and analysis while giving instructors visibility for feedback and review.


### Notes

- Don’t create a PR within your own repo—PRs must be opened against this repository for instructors to see them.

- You can update your PR later by pushing additional commits to your branch.

- If you’re unsure which template to use, ask or consult the assignment brief.

```
        +------------------------+
        | 1. Initiate the Repo   |  ←-- Student clicks "Fork"
        +------------------------+
                  |
                  v
        +--------------------+
        | 2. Do the Work     |  ←-- Edit files, commit code, write report
        +--------------------+
                  |
                  v
        +----------------------------+
        | 3. Open Pull Request       |
        |    - Against original repo |
        |    - Choose correct PR     |
        |      template              |
        | (classification, etc.)     |
        +----------------------------+
                  |
                  v
        +--------------------+
        | 4. Instructors See |
        |    and Review PR   |  ←-- Feedback, comments!
        +--------------------+

```
