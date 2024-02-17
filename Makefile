.PHONY: setup clean

setup:
	poetry install
	poetry shell

clean:
	rm -rf __pycache__

