# BC count
Count red, white blood cells to detect various diseases such as blood cancer (leukemia), lower red blood cells (anemia)...

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Project Structure](#project-structure)
* [Setup](#setup)
* [License](#license)

## Project Structure
```
bc-count/
|-- bc-count/
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
|-- bin/
|   |-- bc-count
|
|-- docs/
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
$ git clone https://github.com/nemo256/bc-count
$ cd bc-count
```
- Activate virtual environment:
```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
- Now just run the script:
```
$ python bc-count/main.py
```

## License
- Please read cell-count/LICENSE.
