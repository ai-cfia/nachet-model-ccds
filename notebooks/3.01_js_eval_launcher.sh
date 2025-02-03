#!/bin/bash
screen -dmS "eval1" -L -Logfile "eval_15spp_6_seed_zoom_120250130.log" bash -c "./notebooks/shell/./4.01_js_evaluate_6_seed_zoom_120250130.sh"
sleep 10
screen -dmS "eval2" -L -Logfile "eval_15spp_2_seed_zoom_120250130.log" bash -c "./notebooks/shell/./4.01_js_evaluate_2_seed_zoom_120250130.sh"
sleep 10
screen -dmS "eval3" -L -Logfile "eval_27spp_120250130.log" bash -c "./notebooks/shell/./4.01_js_evaluate_27spp_120250130.sh"
sleep 10
screen -dmS "eval4" -L -Logfile "eval_27spp_220250130.log" bash -c "./notebooks/shell/./4.01_js_evaluate_27spp_220250130.sh"

# ./notebooks/shell/3.02_js_evaluate_6_seed_zoom_220250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_2_seed_zoom_220250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_6_seed_zoom_320250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_2_seed_zoom_320250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_27spp_320250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_1_seed_zoom_120250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_1_seed_zoom_220250130.sh && \
# ./notebooks/shell/3.02_js_evaluate_1_seed_zoom_320250130.sh
