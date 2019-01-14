#include <stdio.h>
#include <stdlib.h>
#include <handy.h>
#include "test.h"

int main(void) {
    double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1];
    double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1];

    init_genrand(SEED);

    // 描画設定
    HgOpen(TATE, YOKO);
    HgSetFillColor(HG_BLUE);
            
    while (1) {
        load_model(w_mid, w_out);
        test(w_mid, w_out);
    }

    HgClose();

    return (0);
}
