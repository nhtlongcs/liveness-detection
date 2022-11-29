## Contribution guide

If you want to contribute to this repo, please follow steps below:

1. Fork your own version from this repository
1. Checkout to another branch, e.g. `fix-loss`, `add-feat`.
1. Make changes/Add features/Fix bugs
1. Add test cases in the `test` folder and run them to make sure they are all passed (see below)
1. Create and describe feature/bugfix in the PR description (or create new document)
1. Push the commit(s) to your own repository
1. Create a pull request on this repository

```bash
pip install pytest
python -m pytest tests/
```

Expected result:

```bash
============================== test session starts ==============================
platform linux -- Python 3.9.13, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/nhtlong/playground/zalo-ai/liveness-detection
plugins: anyio-3.6.1, order-1.0.1
collected 12 items                                                                                                                                                                                    

tests/test_args.py ...                                                      [ 25%]
tests/test_data.py .                                                        [ 33%]
tests/test_env.py .                                                         [ 41%]
tests/units/test_evaluation.py .                                            [ 50%]
tests/units/test_extractor.py ..                                            [ 66%]
tests/units/test_image_folder_from_csv_ds.py .                              [ 75%]
tests/units/test_model.py .                                                 [ 83%]
tests/units/test_train_and_resume.py .                                      [ 91%]
tests/units/test_load_and_predict.py .                                      [100%]
...
======================= 12 passed, 10 warnings in 30.95s =========================
```

To run code-format

```bash
pip install pre-commit
pre-commit install
```
And every time you commit, the code will be formatted automatically. Or you can run `pre-commit run -a` to format all files.

Expected result:
```bash
$ git add scripts/ cli/
$ git commit -m "rename"         

[WARNING] Unstaged files detected.                                                               
[INFO] Stashing unstaged files to /home/nhtlong/.cache/pre-commit/...
yapf........................................................................Passed
[INFO] Restored changes from /home/nhtlong/.cache/pre-commit/...
[main f552910] rename                                                                            
 4 files changed, 4 insertions(+), 8 deletions(-)                                                
 rename {scripts => cli}/make_soup.py (100%)                                                     
 rename {scripts => cli}/predict.py (91%)                                                        
 rename {scripts => cli}/train.py (97%)                                                          
 rename {scripts => cli}/validate.py (100%)                                  
 ```