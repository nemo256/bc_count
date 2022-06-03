# Cell count
Count red, white blood cells to detect various diseases such as blood cancer (leukemia), lower red blood cells (anemia)...

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Project Structure](#project-structure)
* [Setup](#setup)
* [License](#license)

## Project Structure
```
cell-count/
|-- bin/
|   |-- cell-count
|
|-- docs/
|
|-- cell-count/
|   |-- test/
|   |   |-- __init__.py
|   |   |-- test_data.py
|   |   |-- test_main.py
|   |   |-- test_model.py
|   |   
|   |-- __init__.py
|   |-- data.py
|   |-- main.py
|   |-- model.py
|
|-- AUTHORS
|-- LICENSE
|-- README.md
|-- requirements.txt 
|-- setup.py
|-- TODO.md
```

## Setup

- Download the project:
```
$ git clone https://github.com/nemo256/cell-count
$ cd cell-count
```
- Activate virtual environment:
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
- Now just run the script:
```
$ python bin/cell-count
```

## License
- Please read cell-count/LICENSE.
