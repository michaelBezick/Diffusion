push:
	git add --all
	git commit -m "Edits"
	git push
show:
	squeue -u mbezick
clean:
	rm *.out
interactive:
	#sinteractive -A kildisha-k --nodes=1 --gpus-per-node=1
	#sinteractive --nodes=1 --gpus-per-node=1 --constraint="B|D|K|I"
	sinteractive --nodes=1 --gpus-per-node=1 --time=4:00:00
