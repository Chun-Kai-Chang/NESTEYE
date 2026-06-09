best.pt：訓練後的最佳權重。
best_reparam_v6.pt：使用 best.pt 進行重參數化後的結果。
yolov7_tiny.nb：將 best_reparam_v6.pt 輸入至 AMB82 AI Convert Model後的結果，部署於 AMB82-mini 的 SD 卡內。

部署於 AMB82-mini 的 SD 卡前，需要先開一個名為 NN_MDL 的資料夾，並將 yolov7_tiny.nb 放入資料夾內。

將 best.pt、best_reparam_v6.pt 移至路徑 yolov7/models/ 內。