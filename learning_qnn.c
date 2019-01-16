#include <stdio.h>
#include <handy.h>
#include "learning_qnn.h"

extern int episode;

int main(void) {
    int c;
    int j, t, re;
    int s[5], a, s_next[5], a_next;
    double r;
    int batch_s[BATCH_SIZE][5], batch_a[BATCH_SIZE], batch_s_next[BATCH_SIZE][5];
    double batch_r[BATCH_SIZE];
    int exp_memory_s[BATCH_SIZE][5], exp_memory_a[BATCH_SIZE], exp_memory_s_next[BATCH_SIZE][5];
    double exp_memory_r[BATCH_SIZE];
    double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1];
    double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1];
    double result_mid[MID_UNIT_NO];
    float rewardsum;

    int hit;
    int batch_count;
    int step;
    int test_freq = EPISODE_NO/20;

    print_setting();

    init_genrand(SEED);

    // 描画設定
    HgOpen(TATE, YOKO);
    HgSetFillColor(HG_BLUE);

    // ニューラルネットの重み初期化
    initW(w_mid, w_out);

    // 収益の初期化
    rewardsum = 0.0;

    batch_count = 0;
    step = 0;

    //学習開始
    for (j = 0; j < EPISODE_NO; j++) {
        episode = j;
        // 状態の初期化
        initS(s);

        for (t = 0; t < STEP_NO; t++) {
            // 政策piから行動を決定
            a = pi(s, w_mid, w_out, result_mid, EPSILON);

            // 決定した行動をもとに次の状態へ遷移
            statetransition(s, a, s_next, &hit);

            // 遷移した事による報酬を観測
            r = reword(s_next, hit);
            rewardsum += r;

            // 良い経験を保存
            append_exp_memory(s, a, r, s_next, exp_memory_s, exp_memory_a, exp_memory_r, exp_memory_s_next, step);
            step++;
            // バッチとして保存
            append_batch(s, a, r, s_next, batch_s, batch_a, batch_r, batch_s_next, batch_count);
            batch_count++;

            // バッチが貯まったら学習
            if (batch_count == BATCH_SIZE) {
                // Q値の更新
                learning_units(batch_s, batch_a, batch_r, batch_s_next, w_mid, w_out, result_mid, BATCH_SIZE);
                // 良い経験は消えないように再学習させる
                if (step >= EXP_SIZE) {
                    learning_units(exp_memory_s, exp_memory_a, exp_memory_r, exp_memory_s_next, w_mid, w_out, result_mid, EXP_SIZE);
                }
                batch_count = 0;
            }

            // 状態を観測
            s[BALL_X] = s_next[BALL_X];
            s[BALL_Y] = s_next[BALL_Y];
            s[BALL_VEC_X] = s_next[BALL_VEC_X];
            s[BALL_VEC_Y] = s_next[BALL_VEC_Y];
            s[BAR_X] = s_next[BAR_X];
            
            // ボールを取り逃がしたら終了
            if (s[BALL_Y] < 0) {
                break;
            }
        }

        if (j % 1000 == 0) {
            // 1000エピソー度ごとにprogress_barに報酬平均を表示して学習進捗を確認
            printf("reward_ave=%.5f", rewardsum/1000);
            rewardsum = 0.0;
        }
        if (j % test_freq == test_freq-1) {
            printf("\n");
            save_model(w_mid, w_out);
            for (c = 0; c < 5; c++) {
                test(w_mid, w_out);
            }
        }
    }

    HgClose();

    return 0;
}
