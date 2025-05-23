docker build -t my-tf-gpu .

code .

Ctrl+Shift+P : Remote-Containers: Reopen in Container

ha terminálból akarjuk valamiért elindítani a konténert:
docker run -it --gpus all -v "${PWD}:/app" tf-gpu-dev