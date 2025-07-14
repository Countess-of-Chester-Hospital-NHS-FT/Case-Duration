# Project Template (Py)

This git repository is a template that should be used as the default structure for new projects. It was adapted from [NHSE template](https://github.com/nhsengland/nhse-repository-template/tree/main) and [BSOL template](https://github.com/Birmingham-and-Solihull-ICS/BSOLproject) . It won't fit all circumstances perfectly, and you can adapt it to your needs and make a pull request if you want to suggest changes to the template.

The aim of this template is two-fold: firstly to give a common structure for analytical projects to aid
reproducibility, secondly to allow for additional security settings as default to prevent accidental upload of files that should not be committed to Git and GitHub.

## To use this template, please use the following practises:

* Put any data files in the `data` folder.  This folder is explicitly named in the .gitignore file.  A further layer of security is that all xls, xlsx, csv and pdf files are also explicit ignored in the whole folder as well.  ___If you need to commit one of these files, you must use the `-f` (force) command in `commit`, or specify the file as an exception in the .gitignore but you must be sure there is no identifiable data.__
* Adapt the .gitignore to suit your needs.
* Save any documentation in the `docs` file.  This does not mean you should avoid commenting your code, but if you have an operating procedure or supporting documents, add them to this folder. If any documentation contains sensitive information that shouldn't be pushed, make sure you add the document to the .gitignore file.
* Please save all output: data, formatted tables, graphs etc. in the output folder.  This is also implicitly ignored by git, but you can use the `-f` (force) command in `commit` to add any you wish to publish to github.
* Change the codeowners file (in .github folder) so you are the codeowner.

### Please also consider the following:
* Comment your code to make sure others can follow.
* Consider your naming conventions: we recommend `snake case` where spaces are replaced by underscores and no capitals are use. E.g. `outpatient_referral_data`

__Please update/replace this README file with one relevant to your project__

# Project Name

### Status
This project is currently work in progress / completed / being maintained / not being maintained

### About the project
Short description and purpose / objectives of the project

Link to any refereneces / publications

Note: Only public or dummy data are shared in this repository

### Project Structure

* The main code is found in the root of the repo
* Any notes which may be helpful

### Built with
List of R / python versions and packages or if these are captured in a requirements file

### Outputs
If not already described.

### Contributing
Contributions and identification of issues are welcomed.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/Feature`)
3. Commit your Changes (`git commit -m 'Add some Feature'`)
4. Push to the Branch (`git push origin feature/Feature`)
5. Open a Pull Request

### License
Unless stated otherwise, the codebase is released under the MIT Licence. This covers both the codebase and any sample code in the documentation.

See LICENSE for more information.
