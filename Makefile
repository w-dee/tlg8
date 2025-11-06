.PHONY: sweep-reorder sweep-view

sweep-reorder:
	python3 tools/sweep_reorder.py

sweep-view:
	@echo "Open runs/sweeps/reorder_v1/results.md"
