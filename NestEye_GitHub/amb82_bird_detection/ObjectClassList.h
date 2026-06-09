#ifndef OBJECT_CLASS_LIST_H
#define OBJECT_CLASS_LIST_H

typedef struct ObjectClassItem {
    const char *objectName;
    int filter;
} ObjectClassItem;

// class ID 必須和 YOLO 訓練時的 classes.txt / data.yaml 順序完全一致
ObjectClassItem itemList[3] = {
    {"Columba_livia", 1},       // 0: 原鴿
    {"Passer_montanus", 1},    // 1: 麻雀
    {"Hirundo_rustica", 1}     // 2: 家燕
};

#endif