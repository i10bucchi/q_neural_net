
/*定数定義*/

/*モデル*/
#define TATE 200
#define YOKO 200
#define ACTION_NO 3
#define ACTION_LEFT 0
#define ACTION_RIGHT 1
#define ACTION_STOP 2

#define BAR_LENGTH 60
#define BAR_SPEED 3
#define SPEED_LIMIT 3

/*状態のインデックス*/
#define STATE_NO 5
#define BALL_X 0
#define BALL_Y 1
#define BALL_VEC_X 2
#define BALL_VEC_Y 3
#define BAR_X 4

#define SEED 3

#define LINE_NUM 256

/*強化学習*/
#define ALPHA 0.08
#define EPSILON 0.3
#define GAMMA 0.9
#define EPISODE_NO 200000
#define STEP_NO 10000

/*ニューラルネット*/
#define NNLIMIT 0.00006
#define NNALPHA 0.01
// #define INPUT_UNIT_NO (YOKO + TATE + SPEED_LIMIT*2 + 2 + YOKO - BAR_LENGTH)
#define INPUT_UNIT_NO STATE_NO
#define MID_UNIT_NO 96
#define OUTPUT_UNIT_NO ACTION_NO
#define BATCH_SIZE 100
#define EXP_SIZE BATCH_SIZE
#define LOOP_LIMIT 500