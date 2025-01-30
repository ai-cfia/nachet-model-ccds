#!/bin/bash

screen -dmS "train1" bash -c "./shell/3.02_js_train_15spp_6_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_train_15spp_6_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_train_15spp_6_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_train_15spp_2_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_train_15spp_2_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_train_15spp_2_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_train_15spp_1_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_train_15spp_1_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_train_15spp_1_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_train_27spp_6_seed_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_train_27spp_6_seed_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_train_27spp_6_seed_320250130.sh"