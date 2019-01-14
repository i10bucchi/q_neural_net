#include "const.h"

// MT.c
extern void init_genrand(unsigned long s);

// helper.h
extern void initS(int s[]);
extern void load_model(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);