test-causalqa-lin:
	cd causalqa ; datasets-cli test causalqa.py --save_info --all_configs

test-causalqa-win:
	cd causalqa & datasets-cli test causalqa.py --save_info --all_configs