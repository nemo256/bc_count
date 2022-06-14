# The production version is at [nemo256/cbc](https://github.com/nemo256/cbc)
# Blood Cells Count
Count red, white blood cells to detect various diseases such as blood cancer (leukemia), lower red blood cells count (anemia)...

![Sample](sample.png)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Project Structure](#project-structure)
* [Develop](#develop)
* [License](#license)

## Project Structure
```
bc-count/
|-- bc-count/
|   |-- config.py
|   |-- data.py
|   |-- main.py
|   |-- model.py
|
|-- bin/
|   |-- bc-count
|
|-- docs/
|
|-- data/
|   |-- plt/
|   |-- rbc/
|   |-- wbc/
|
|-- models/

|-- output/
|   |-- plt/
|   |-- rbc/
|   |-- wbc/
|
|-- AUTHORS
|-- LICENSE
|-- README.md
|-- TODO.md
|-- requirements.txt 
|-- setup.py
```

## Develop
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
- Now just adapt the code to your need and then run using the command:
```
$ python bc-count/main.py
```

## License
- Please read cell-count/LICENSE.
