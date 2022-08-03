''' 
Args:
--datasets: choosing datsets
--expType: experiments type 
--expocode: do determine output path {/data/result/[datasets]/[expType]/[expCode]}
'''

# for Twitter dataset

python main.py --datasets 'Twitter' --expType 'vis' --expCode 'vis_epoch50_freeze' 

python main.py --datasets 'Twitter' --expType 'text' --expCode 'text_epoch50_freeze' 

python main.py --datasets 'Twitter' --expType 'wo_fusion' --expCode 'wo_fusion_epoch50_freeze' 

python main.py --datasets 'Twitter' --expType 'mulT' --expCode 'mulT_epoch50_freeze2'  

python main.py --datasets 'Twitter' --expType 'all' --expCode 'all_epoch50_freeze2'

# for Weibo Dataset

python main.py --datasets 'Weibo' --expType 'vis' --expCode 'vis_epoch50_freeze'  

python main.py --datasets 'Weibo' --expType 'text' --expCode 'text_epoch50_freeze'  

python main.py --datasets 'Weibo' --expType 'wo_fusion' --expCode 'wo_fusion_epoch50_freeze'  

python main.py --datasets 'Weibo' --expType 'mulT' --expCode 'mulT_epoch50_freeze'  

python main.py --datasets 'Weibo' --expType 'all' --expCode 'all_epoch50_freeze2'