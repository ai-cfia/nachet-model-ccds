#!/bin/bash

# screen -dmS "train1" bash -c "./notebooks/shell/./3.02_js_train_15spp_6_seed_zoom_120250130.sh"
# sleep 60
# screen -dmS "train2" bash -c "./notebooks/shell/./3.02_js_train_15spp_2_seed_zoom_120250130.sh"
# sleep 60

# screen -dmS "train3" bash -c "./notebooks/shell/./3.02_js_train_27spp_6_seed_120250130.sh"
# sleep 60
# screen -dmS "train3" bash -c "./notebooks/shell/./3.02_js_train_27spp_6_seed_220250130.sh"
# screen -dmS "train3" bash -c "./notebooks/shell/3.02_js_train_27spp_6_seed_320250130.sh"

# screen -dmS "train1" bash -c "./notebooks/shell/3.02_js_train_15spp_6_seed_zoom_320250130.sh"
# sleep 60
# screen -dmS "train2" bash -c "./notebooks/shell/3.02_js_train_15spp_2_seed_zoom_320250130.sh"

# sleep 60
# screen -dmS "train2" bash -c "./notebooks/shell/3.02_js_train_15spp_2_seed_zoom_220250130.sh"

screen -dmS "train1" -L -Logfile "15spp_6_seed_zoom_220250130.log" bash -c "./notebooks/shell/3.02_js_train_15spp_6_seed_zoom_220250130.sh"
sleep 60
screen -dmS "train2" -L -Logfile "15spp_1_seed_zoom_120250130.log" bash -c "./notebooks/shell/3.02_js_train_15spp_1_seed_zoom_120250130.sh"

# screen -dmS "train2" bash -c "./notebooks/shell/3.02_js_train_15spp_1_seed_zoom_220250130.sh"
# sleep 60
# screen -dmS "train3" bash -c "./notebooks/shell/3.02_js_train_15spp_1_seed_zoom_320250130.sh"
