push:
	git add --all
	git commit -m "Edits"
	git push
show:
	squeue -u mbezick
clean:
	rm *.out
interactive:
	sinteractive --nodes=1 --gpus-per-node=1 --constraint="K|I"
