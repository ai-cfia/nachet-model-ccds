#!/bin/bash

screen -dmS "train1" bash -c "./shell/3.02_js_evaluate_6_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_evaluate_6_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_evaluate_6_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_evaluate_2_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_evaluate_2_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_evaluate_2_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_evaluate_1_seed_zoom_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_evaluate_1_seed_zoom_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_evaluate_1_seed_zoom_320250130.sh"

screen -dmS "train1" bash -c "./shell/3.02_js_evaluate_27spp_120250130.sh"
screen -dmS "train2" bash -c "./shell/3.02_js_evaluate_27spp_220250130.sh"
screen -dmS "train3" bash -c "./shell/3.02_js_evaluate_27spp_320250130.sh"