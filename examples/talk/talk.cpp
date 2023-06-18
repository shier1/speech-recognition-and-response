// Talk with AI
//

#include "common.h"
#include "common-sdl.h"
#include "whisper.h"
// #include <unistd.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <random>

#include "model.h"

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool print_special = false;
    bool print_energy  = false;
    bool no_timestamps = true;

    std::string person    = "Santa";
    std::string language  = "en";
    std::string model_wsp = "models/ggml-base.en.bin";
    std::string model_gpt = "models/ggml-gpt-2-117M.bin";
    std::string speak     = "./examples/talk/speak";
    std::string fname_out;
};

struct app_params {
    int32_t seed = -1;  // RNG seed
    int32_t n_threads =
            std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_predict = 128;     // new tokens to predict
    int32_t repeat_last_n = 64;  // last n tokens to penalize
    int32_t n_ctx = 2048;        // context size

    // sampling parameters
    int32_t top_k = 40;
    float top_p = 0.95f;
    float temp = 0.10f;
    float repeat_penalty = 1.30f;

    std::string model ;  // model path

    bool use_color = true;  // use color to distinguish generations and inputs
    bool use_mmap = false;  // use mmap to load model
    std::string dtype = "float32";  // configure the compute dtype
    std::string mtype = "chatglm";  // the model type name, llama
};


void print_usage(int /*argc*/, char ** argv, const whisper_params & params, const app_params& chatparams);

bool params_parse(int argc, char** argv, whisper_params& params, app_params& chatparams) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--seed") {
            chatparams.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            chatparams.n_threads = std::stoi(argv[++i]);
            // params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            chatparams.top_k = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            chatparams.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-d" || arg == "--dtype") {
            chatparams.dtype = argv[++i];
        } else if (arg == "--top_p") {
            chatparams.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            chatparams.temp = std::stof(argv[++i]);
        } else if (arg == "--repeat_last_n") {
            chatparams.repeat_last_n = std::stoi(argv[++i]);
        } else if (arg == "--repeat_penalty") {
            chatparams.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            chatparams.model = argv[++i];
        } else if (arg == "--color") {
            chatparams.use_color = true;
        } else if (arg == "--mmap") {
            chatparams.use_mmap = true;
        }else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params, chatparams);
            exit(0);
        }
        else if (arg == "-vms" || arg == "--voice-ms")      { params.voice_ms      = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-pe"  || arg == "--print-energy")  { params.print_energy  = true; }
        else if (arg == "-p"   || arg == "--person")        { params.person        = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper") { params.model_wsp     = argv[++i]; }
        else if (arg == "-s"   || arg == "--speak")         { params.speak         = argv[++i]; }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params, chatparams);
            exit(0);
        }

    }
    return true;
}

void print_usage(int /*argc*/, char ** argv, const whisper_params & params, const app_params & chatparams) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N    [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",     params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy  [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -p NAME,  --person NAME   [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT    [%-7s] command for TTS\n",                             params.speak.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");

    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", chatparams.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", chatparams.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", chatparams.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", chatparams.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", chatparams.n_ctx);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", chatparams.temp);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", chatparams.model.c_str());
    fprintf(stderr, "  --mmap                enable mmap when read weights, default = false\n");
    fprintf(stderr, "  -d type               configure the compute type, default float32, can be float32 and flot16 now.\n");
    fprintf(stderr, "  --model_type type     the model type name, default llama, can only be llama now.\n");
    fprintf(stderr, "\n");
}

std::string transcribe(whisper_context * ctx, const whisper_params & params, const std::vector<float> & pcmf32, float & prob, int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.audio_ctx        = params.audio_ctx;
    wparams.speed_up         = params.speed_up;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

const std::string k_prompt =
R"(This is a dialogue between {0} (A) and a person (B). The dialogue so far is:

B: Hello {0}, how are you?
A: I'm fine, thank you.
{1}
Here is how {0} (A) continues the dialogue:

A:)";



int main(int argc, char ** argv) {
    whisper_params params;
    app_params chat_params;

    if (params_parse(argc, argv, params, chat_params) == false) {
        return 1;
    }

    if (chat_params.seed < 0) {
        chat_params.seed = time(NULL);
    }

    if (whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        exit(0);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, chat_params.seed);
    // whisper init
    struct whisper_context * ctx_wsp = whisper_init_from_file(params.model_wsp.c_str());

    // chat init
    inferllm::ModelConfig config;
    config.compt_type = chat_params.dtype;
    config.nr_thread = chat_params.n_threads;
    config.enable_mmap = chat_params.use_mmap;
    config.nr_ctx = chat_params.n_ctx;
    std::shared_ptr<inferllm::Model> model =
            std::make_shared<inferllm::Model>(config, chat_params.mtype);
    model->load(chat_params.model);
    model->init(chat_params.top_k, chat_params.top_p, chat_params.temp, chat_params.repeat_penalty,
    chat_params.repeat_last_n, chat_params.seed);


    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
        
        fprintf(stderr, "%s: interactive mode on.\n", __func__);
        fprintf(stderr,
            "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
            "repeat_last_n = %i, repeat_penalty = %f\n",
            chat_params.temp, chat_params.top_k, chat_params.top_p, chat_params.repeat_last_n,
            chat_params.repeat_penalty);
        fprintf(stderr, "\n\n");
        fprintf(stderr,
            "== 运行模型中. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || \
        defined(_WIN32)
            " - 输入 Ctrl+C 将在推出程序.\n"
#endif
            " - 如果你想换行，请在行末输入'\\'符号.\n");
    }


    // init audio
    audio_async audio(30*1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    bool is_running  = true;
    bool force_speak = false;

    float prob0 = 0.0f;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    // prompt user immediately after the starting prompt has been loaded
    auto fix_word = [](std::string& word) {
        auto ret = word;
        if (word == "<n>" || word == "<n><n>")
            word = "\n";
        if (word == "<|tab|>")
            word = "\t";
        int pos = word.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(word.substr(8, word.size() - 10).c_str());
            word = std::string(space_num, ' ');
        }
        pos = word.find("▁");
        if (pos != -1) {
            word.replace(pos, pos + 3, " ");
        }
        // Fix utf-8 garbled characters
        if (word.length() == 6 && word[0] == '<' &&
            word[word.length() - 1] == '>' && word[1] == '0' &&
            word[2] == 'x') {
            int num = std::stoi(word.substr(3, 2), nullptr, 16);
            word = static_cast<char>(num);
        }
    };

    // main loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int64_t t_ms = 0;
        {
            audio.get(2000, pcmf32_cur);

            if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy) || force_speak) {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                audio.get(params.voice_ms, pcmf32_cur);

                std::string text_heard;

                if (!force_speak) {
                    text_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prob0, t_ms));
                }
                printf("Master: %s\n", text_heard.c_str());

                if (text_heard.empty() || force_speak) {
                    printf("nothing hear skipped");
                    audio.clear();
                    continue;
                }
                force_speak = false;

                std::string text_to_speak;
                {
                    text_heard = text_heard + "\n";
                    // the chat glm response
                    if (!text_heard.empty()){
                        int token;
                        text_to_speak = model->decode(text_heard, token);
                        fix_word(text_to_speak);
                        text_to_speak = "回复：" + text_to_speak;
                        printf("%s", text_to_speak.c_str());
                        fflush(stdout);
                    }
                    while(true){
                        int token;
                        auto o = model->decode_iter(token);
                        fix_word(o);
                        text_to_speak += o;
                        printf("%s", o.c_str());
                        fflush(stdout);
                        if (token == 130005) {
                            printf("\n");
                            break;
                        }
                    }
                    // system((params.speak + " " + " \"" + text_to_speak + "\"").c_str());
//                    pause();
                }
                audio.clear();
            }
        }
    }

    audio.pause();
    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    return 0;
}
