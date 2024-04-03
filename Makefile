push:
	git add --all
	git commit -m "Edits"
	git push
show:
	squeue -u mbezick
clean:
	rm *.out
interactive:
	sinteractive -A standby -N1 -n1 --gpus-per-node=1
