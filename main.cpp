#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "HeartsEnv.hpp"
#include "SearchPlayer.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

std::string CardToString(int card_id) {
    if (card_id < 0 || card_id > 51) return "??";
    const char* suits[] = {"\xE2\x99\xA3", "\xE2\x99\xA6", "\xE2\x99\xA0", "\xE2\x99\xA5"};
    const char* ranks[] = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    int suit = card_id / 13;
    int rank = card_id % 13;
    return std::string(ranks[rank]) + suits[suit];
}

const char* PASS_DIR_NAMES[] = {"LEFT", "RIGHT", "ACROSS", "HOLD"};

// Seat offset a pass travels: receiver = (giver + offset) % 4
int PassOffset(int dir) {
    return dir == 0 ? 1 : (dir == 1 ? 3 : (dir == 2 ? 2 : 0));
}

class CardButton : public ComponentBase {
public:
    std::string label;
    std::function<void()> on_click;
    bool is_legal;
    Suit suit;
    std::function<bool()> is_selected; // set during the passing phase
    bool hovered = false;
    Box box_; // For mouse collision detection

    CardButton(std::string label_, std::function<void()> on_click_, bool is_legal_, Suit suit_,
               std::function<bool()> is_selected_ = nullptr)
        : label(std::move(label_)), on_click(std::move(on_click_)), is_legal(is_legal_),
          suit(suit_), is_selected(std::move(is_selected_)) {}

    Element Render() override {
        bool selected = is_selected && is_selected();
        auto t = text((selected ? "*" : " ") + label + (selected ? "*" : " ")) | border;

        bool is_focused = Focused();
        if (hovered || is_focused) t = t | inverted;
        if (selected) t = t | bgcolor(Color::Blue) | bold;

        if (!is_legal) {
            if (suit == Suit::Hearts || suit == Suit::Diamonds) {
                t = t | dim | color(Color::DarkRed);
            } else {
                t = t | dim | color(Color::GrayDark);
            }
        } else if (suit == Suit::Hearts || suit == Suit::Diamonds) {
            t = t | color(Color::Red);
        } else {
            t = t | color(Color::White);
        }
        
        return t | reflect(box_); // Capture layout coordinates
    }

    bool Focusable() const override { return is_legal; }

    bool OnEvent(Event e) override {
        if (e.is_mouse()) {
            bool was_hovered = hovered;
            hovered = box_.Contain(e.mouse().x, e.mouse().y);
            
            if (hovered && !was_hovered && is_legal) {
                TakeFocus(); // Automatically steal keyboard focus!
            }
            
            if (hovered && e.mouse().button == Mouse::Left && e.mouse().motion == Mouse::Released) {
                if (is_legal) on_click();
                return true;
            }
        }
        if (e == Event::Return && Focused()) {
            if (is_legal) on_click();
            return true;
        }
        return false;
    }
};

Component MakeCardButton(std::string label, std::function<void()> on_click, bool is_legal, Suit suit,
                         std::function<bool()> is_selected = nullptr) {
    return Make<CardButton>(std::move(label), std::move(on_click), is_legal, suit, std::move(is_selected));
}

// Append one CSV row per completed round (and a MATCH_END row when someone
// crosses 100) so human-vs-AI calibration sessions leave an analyzable record.
// Columns: timestamp,event,round_p0..p3,total_p0..p3[,winner]
static void AppendMatchLog(const std::string& prefix, const GameState& state, bool match_over) {
    std::ofstream log(prefix + "hearts_match_log.csv", std::ios::app);
    if (!log) return;
    if (log.tellp() == 0) {
        log << "timestamp,event,round_p0,round_p1,round_p2,round_p3,"
               "total_p0,total_p1,total_p2,total_p3,winner\n";
    }
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    log << buf << "," << (match_over ? "MATCH_END" : "round");
    for (int i = 0; i < 4; ++i) log << "," << state.round_scores[i];
    for (int i = 0; i < 4; ++i) log << "," << state.total_scores[i];
    if (match_over) {
        int w = 0;
        for (int i = 1; i < 4; ++i) {
            if (state.total_scores[i] < state.total_scores[w]) w = i;
        }
        log << ",P" << w;
    } else {
        log << ",";
    }
    log << "\n";
}

int main() {
    // 1. Load the Model. Try the working directory first (running from the
    // repo root), then two levels up (double-clicking the exe in build/Release).
    torch::jit::script::Module ai_model;
    bool model_loaded = false;
    std::string data_prefix;  // "" (repo root cwd) or "../../" (exe double-clicked)
    for (const char* path : {"hearts_ai_grandmaster.pt", "../../hearts_ai_grandmaster.pt"}) {
        try {
            ai_model = torch::jit::load(path);
            ai_model.eval();
            model_loaded = true;
            data_prefix = (path[0] == '.') ? "../../" : "";
            break;
        } catch (const c10::Error&) {
            // try the next candidate path
        }
    }
    if (!model_loaded) {
        std::cerr << "Error: could not find hearts_ai_grandmaster.pt (looked in the "
                     "current directory and ../../).\n";
        return -1;
    }

    // The deployed model may come from an older lineage with a narrower
    // observation. The observation layout is prefix-stable across versions,
    // so probe which width this model expects and feed exactly that prefix.
    int model_obs_dim = 0;
    for (int dim : {550, 238, 181}) {
        try {
            std::vector<torch::jit::IValue> probe;
            probe.push_back(torch::zeros({1, dim}, torch::kFloat32));
            probe.push_back(torch::ones({1, 52}, torch::kBool));
            ai_model.forward(probe);
            model_obs_dim = dim;
            break;
        } catch (const std::exception&) {
            // try the next known width
        }
    }
    if (model_obs_dim == 0) {
        std::cerr << "Error: model rejected all known observation widths.\n";
        return -1;
    }

    // Optional decision-time search opponent: if the 3-output search trace is
    // available, AI seats search over belief-weighted determinizations instead
    // of playing the raw policy. Falls back silently to the raw policy.
    std::unique_ptr<SearchPlayer> search_ai;
    for (const char* path : {"hearts_ai_search.pt", "../../hearts_ai_search.pt"}) {
        try {
            torch::jit::script::Module sm = torch::jit::load(path);
            sm.eval();
            int sdim = ProbeObsDim(sm);
            if (sdim != 0) {
                SearchPlayer::Config scfg;
                // K=64 is the measured strength plateau (128/256 gain nothing);
                // on the GPU a K=64 decision is ~0.1s. CPU fallback keeps K=24
                // so moves stay responsive without CUDA.
                if (torch::cuda::is_available()) {
                    scfg.device = torch::kCUDA;
                    scfg.determinizations = 64;
                } else {
                    scfg.determinizations = 24;
                }
                scfg.seed = 777;
                scfg.pass_search = true;  // AI opponents also search their passes
                search_ai = std::make_unique<SearchPlayer>(std::move(sm), sdim, scfg);
            }
            break;
        } catch (const c10::Error&) {
            // try the next candidate path
        }
    }

    // 2. Initialize Environment (card passing enabled: Left, Right, Across, Hold)
    std::random_device rd;
    HeartsEnv env(rd(), true);
    env.Reset();
    
    auto screen = ScreenInteractive::Fullscreen();
    
    std::atomic<bool> game_over(false);
    std::atomic<bool> overall_game_over(false);
    std::atomic<bool> ui_ready(false);
    std::atomic<bool> show_trick_override(false);
    std::atomic<bool> god_mode(false);
    std::atomic<int> action_to_play(-1); // Thread synchronization variable
    std::vector<PlayedCard> trick_override;

    // Passing-phase state (shared between UI and game thread)
    std::mutex pass_mutex;
    std::vector<int> pass_selection;   // cards the human has toggled to pass
    std::vector<int> human_kept;       // hand snapshot after the human's picks left
    std::vector<int> received_cards;   // what arrived in the distribution
    std::atomic<bool> pass_confirmed(false);

    // 3. Human UI Component (Interactive Container)
    auto hand_container = Container::Horizontal({});
    
    auto new_game_button = Button("New Game", [&] {
        overall_game_over = false;
        // game_thread will notice and reset
    });
    
    auto game_over_container = Container::Vertical({
        new_game_button
    });
    
    int tab_index = 0;
    auto main_container = Container::Tab({
        hand_container,
        game_over_container
    }, &tab_index);

    auto renderer = Renderer(main_container, [&] {
        ui_ready = true; // first frame proves the event loop is installed
        const auto& state = env.GetState();
        
        if (overall_game_over) {
            tab_index = 1;
            int winner = 0;
            int lowest_score = 9999;
            for (int i = 0; i < 4; ++i) {
                if (state.total_scores[i] < lowest_score) {
                    lowest_score = state.total_scores[i];
                    winner = i;
                }
            }
            std::string winner_text = (winner == 0) ? "You Win!" : "Player " + std::to_string(winner) + " Wins!";
            return vbox({
                text(""),
                text("=================================") | center | bold,
                text("           GAME OVER             ") | center | bold | color(Color::Red),
                text("=================================") | center | bold,
                text(""),
                text(winner_text) | center | bold | color(Color::Green),
                text("Lowest Score: " + std::to_string(lowest_score)) | center,
                text(""),
                text("Final Scores:") | center | bold,
                text("Player 0 (You): " + std::to_string(state.total_scores[0])) | center,
                text("Player 1 (AI):  " + std::to_string(state.total_scores[1])) | center,
                text("Player 2 (AI):  " + std::to_string(state.total_scores[2])) | center,
                text("Player 3 (AI):  " + std::to_string(state.total_scores[3])) | center,
                text(""),
                new_game_button->Render() | center,
                text(""),
                text("Press Ctrl+C to exit.") | center | dim
            }) | center;
        } else {
            tab_index = 0;
        }

        auto score_box = window(text(" Scores [Press 'G' to toggle God Mode] "), vbox({
            text("Player 0 (You): Total " + std::to_string(state.total_scores[0]) + " (Round: " + std::to_string(state.round_scores[0]) + ")"),
            text("Player 1 (AI):  Total " + std::to_string(state.total_scores[1]) + " (Round: " + std::to_string(state.round_scores[1]) + ")"),
            text("Player 2 (AI):  Total " + std::to_string(state.total_scores[2]) + " (Round: " + std::to_string(state.round_scores[2]) + ")"),
            text("Player 3 (AI):  Total " + std::to_string(state.total_scores[3]) + " (Round: " + std::to_string(state.round_scores[3]) + ")"),
            text("This round passes: " + std::string(PASS_DIR_NAMES[env.GetPassDirection()])) | color(Color::Cyan),
        }));
        
        Elements last_trick_elements;
        int last_winner = state.last_trick_winner;
        for (size_t i = 0; i < state.last_trick.size(); ++i) {
            const auto& card = state.last_trick[i];
            int card_id = static_cast<int>(card.card.suit) * 13 + (card.card.rank - 2);
            auto t = text("P" + std::to_string(card.player_id) + ": " + CardToString(card_id));
            if (card.player_id == last_winner) {
                t = t | color(Color::Green) | bold;
            }
            last_trick_elements.push_back(t);
        }
        if (last_trick_elements.empty()) {
            last_trick_elements.push_back(text("None"));
        }
        auto last_trick_box = window(text(" Last Trick "), vbox(std::move(last_trick_elements)) | center);
        
        Elements trick_elements;
        const auto& active_trick = show_trick_override ? trick_override : state.current_trick;
        
        int highest_rank = -1;
        int winning_idx = -1;
        if (active_trick.size() == 4) {
            Suit led_suit = active_trick[0].card.suit;
            for (size_t i = 0; i < active_trick.size(); ++i) {
                if (active_trick[i].card.suit == led_suit && active_trick[i].card.rank > highest_rank) {
                    highest_rank = active_trick[i].card.rank;
                    winning_idx = i;
                }
            }
        }
        
        for (size_t i = 0; i < active_trick.size(); ++i) {
            const auto& card = active_trick[i];
            int card_id = static_cast<int>(card.card.suit) * 13 + (card.card.rank - 2);
            auto t = text("P" + std::to_string(card.player_id) + ": " + CardToString(card_id));
            if (i == winning_idx) {
                t = t | color(Color::Red) | blink | bold;
            }
            trick_elements.push_back(t);
        }
        if (trick_elements.empty()) {
            trick_elements.push_back(text("Waiting for lead..."));
        }
        
        auto trick_box = window(text(" Current Trick "), vbox(std::move(trick_elements)) | center);
        
        Element god_mode_panel;
        if (god_mode) {
            Elements ai_hands_ui;
            for (int i = 1; i <= 3; ++i) {
                Elements cards_ui;
                for (const auto& c : state.hands[i]) {
                    int card_id = static_cast<int>(c.suit) * 13 + (c.rank - 2);
                    auto t = text(" " + CardToString(card_id) + " ") | border;
                    if (c.suit == Suit::Hearts || c.suit == Suit::Diamonds) {
                        t = t | color(Color::Red);
                    }
                    cards_ui.push_back(t);
                }
                ai_hands_ui.push_back(
                    window(text(" Player " + std::to_string(i) + " "), hbox(std::move(cards_ui)) | center)
                );
            }
            god_mode_panel = window(text(" God Mode (AI Hands) "), vbox(std::move(ai_hands_ui)));
        } else {
            god_mode_panel = filler();
        }
        
        int current_player = env.GetCurrentPlayer();
        
        Element status_text;
        if (game_over) {
            status_text = text("Round Over! Dealing next hand...") | center | bold;
        } else if (env.IsPassing()) {
            if (current_player == 0) {
                size_t n_sel;
                {
                    std::lock_guard<std::mutex> lk(pass_mutex);
                    n_sel = pass_selection.size();
                }
                int target = PassOffset(env.GetPassDirection()) % 4;
                status_text = text("Pass 3 cards " + std::string(PASS_DIR_NAMES[env.GetPassDirection()]) +
                                   " to Player " + std::to_string(target) +
                                   "  (" + std::to_string(n_sel) + "/3 selected - click to toggle, then Confirm)")
                              | center | bold | color(Color::Yellow);
            } else {
                status_text = text("Opponents are choosing their passes...") | center | bold;
            }
        } else if (current_player == 0) {
            status_text = text("Your Turn (Select a Card to Play)") | center | bold;
        } else {
            status_text = text("AI is thinking... Player " + std::to_string(current_player) + "'s turn.") | center | bold;
        }

        // Reveal what the pass brought in, for the duration of the first trick
        Element received_box = filler() | size(HEIGHT, EQUAL, 0);
        {
            std::lock_guard<std::mutex> lk(pass_mutex);
            if (!received_cards.empty() && !env.IsPassing() && state.tricks_played == 0) {
                std::string recv = "You received:";
                for (int id : received_cards) recv += " " + CardToString(id);
                int giver = (4 - PassOffset(env.GetPassDirection())) % 4;
                recv += "  (from Player " + std::to_string(giver) + ")";
                received_box = text(recv) | center | color(Color::Green);
            }
        }

        auto bottom_panel = window(text(" Your Hand "), hand_container->Render() | center) | size(HEIGHT, EQUAL, 5);

        return vbox({
            score_box,
            last_trick_box,
            filler(),
            trick_box,
            god_mode_panel,
            status_text,
            received_box,
            bottom_panel
        });
    });

    // 4. Game Logic Thread
    std::thread game_thread([&]() {
        // ScreenInteractive::Post() silently drops tasks until Loop() has
        // installed the event queue. Wait for the first rendered frame so the
        // initial hand build (the human passes first!) cannot be lost.
        while (!ui_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (true) {
            if (overall_game_over) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            if (game_over) {
                game_over = false;
                {
                    std::lock_guard<std::mutex> lk(pass_mutex);
                    pass_selection.clear();
                    human_kept.clear();
                    received_cards.clear();
                }
                pass_confirmed = false;
                env.Reset();
                screen.PostEvent(Event::Custom);
            }
            
            int current_player = env.GetCurrentPlayer();
            auto legal_actions = env.GetLegalActions();
            bool is_passing = env.IsPassing();

            std::vector<Card> current_hand = env.GetState().hands[0];

            if (is_passing && current_player == 0) pass_confirmed = false;

            screen.Post([&, current_player, legal_actions, current_hand, is_passing]() {
                hand_container->DetachAllChildren();
                for (const auto& c : current_hand) {
                    int action_id = (static_cast<int>(c.suit) * 13) + (c.rank - 2);

                    bool is_legal = false;
                    if (current_player == 0) {
                        for (int i = 0; i < 13; ++i) {
                            if (legal_actions[i] == action_id) {
                                is_legal = true;
                                break;
                            }
                        }
                    }

                    Component btn;
                    if (is_passing && current_player == 0) {
                        // Passing: clicks toggle selection (max 3, click again to remove)
                        btn = MakeCardButton(CardToString(action_id), [&, action_id]() {
                            std::lock_guard<std::mutex> lk(pass_mutex);
                            auto it = std::find(pass_selection.begin(), pass_selection.end(), action_id);
                            if (it != pass_selection.end()) {
                                pass_selection.erase(it);
                            } else if (pass_selection.size() < 3) {
                                pass_selection.push_back(action_id);
                            }
                            screen.PostEvent(Event::Custom);
                        }, is_legal, c.suit, [&, action_id]() {
                            std::lock_guard<std::mutex> lk(pass_mutex);
                            return std::find(pass_selection.begin(), pass_selection.end(), action_id) != pass_selection.end();
                        });
                    } else {
                        btn = MakeCardButton(CardToString(action_id), [&, action_id]() {
                            if (action_to_play == -1) action_to_play = action_id;
                        }, is_legal, c.suit);
                    }

                    hand_container->Add(btn);
                }
                if (is_passing && current_player == 0) {
                    hand_container->Add(Button(" Confirm Pass ", [&] {
                        std::lock_guard<std::mutex> lk(pass_mutex);
                        if (pass_selection.size() == 3) pass_confirmed = true;
                    }));
                }
            });

            screen.PostEvent(Event::Custom); // Force redraw

            if (current_player == 0 && is_passing) {
                // Wait for the human to select 3 cards and confirm, then submit
                while (!pass_confirmed && !game_over && !overall_game_over) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (game_over || overall_game_over) continue;

                std::vector<int> picks;
                {
                    std::lock_guard<std::mutex> lk(pass_mutex);
                    picks = pass_selection;
                }
                for (int id : picks) {
                    env.Step(id);
                }
                {
                    // Snapshot the 10 cards kept; the distribution delta = received
                    std::lock_guard<std::mutex> lk(pass_mutex);
                    human_kept.clear();
                    for (const auto& c : env.GetState().hands[0]) {
                        human_kept.push_back((static_cast<int>(c.suit) * 13) + (c.rank - 2));
                    }
                }
                screen.PostEvent(Event::Custom);
                continue; // loop re-enters for the AI players' passes
            }

            if (current_player == 0) {
                action_to_play = -1; // Reset action signal
                
                while (action_to_play == -1 && !game_over) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                
                if (game_over) break;
                
                int chosen_action = action_to_play;
                
                if (env.GetState().current_trick.size() == 3) {
                    std::vector<Card> pre_pause_hand = env.GetState().hands[0];
                    screen.Post([&, chosen_action, pre_pause_hand]() {
                        hand_container->DetachAllChildren();
                        for (const auto& c : pre_pause_hand) {
                            int action_id = (static_cast<int>(c.suit) * 13) + (c.rank - 2);
                            if (action_id == chosen_action) continue; 
                            
                            auto btn = MakeCardButton(CardToString(action_id), [](){}, false, c.suit);
                            hand_container->Add(btn);
                        }
                    });
                    
                    trick_override = env.GetState().current_trick;
                    Card c;
                    c.suit = static_cast<Suit>(chosen_action / 13);
                    c.rank = (chosen_action % 13) + 2;
                    trick_override.push_back({0, c});
                    show_trick_override = true;
                    screen.PostEvent(Event::Custom);
                    
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    show_trick_override = false;
                }
                
                auto result = env.Step(chosen_action);
                if (result.done) {
                    screen.PostEvent(Event::Custom); // Draw the final trick and scores
                    std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 second pause
                    
                    bool hit_100 = false;
                    for (int i = 0; i < 4; ++i) {
                        if (env.GetState().total_scores[i] >= 100) hit_100 = true;
                    }
                    AppendMatchLog(data_prefix, env.GetState(), hit_100);
                    if (hit_100) {
                        overall_game_over = true;
                        game_over = true;
                        screen.PostEvent(Event::Custom);
                        continue;
                    } else {
                        {
                            std::lock_guard<std::mutex> lk(pass_mutex);
                            pass_selection.clear();
                            human_kept.clear();
                            received_cards.clear();
                        }
                        pass_confirmed = false;
                        env.Reset(); // Deal new round
                        screen.PostEvent(Event::Custom);
                        continue;
                    }
                }
                screen.PostEvent(Event::Custom);
                
            } else {
                // AI Turn
                int action_id;
                if (search_ai) {
                    // Decision-time search: its compute time IS the thinking pause
                    try {
                        action_id = search_ai->ChooseAction(env);
                    } catch (const std::exception& e) {
                        std::ofstream out("crash.txt");
                        out << "Exception in search: " << e.what() << std::endl;
                        exit(1);
                    }
                    if (is_passing) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                } else {
                    auto obs = env.Observe();

                    torch::Tensor obs_tensor = torch::from_blob(obs.data(), {1, model_obs_dim}, torch::kFloat32).clone();

                    torch::Tensor mask_tensor = torch::zeros({1, 52}, torch::kBool);
                    bool* mask_ptr = mask_tensor.data_ptr<bool>();
                    for(int i = 0; i < 13; ++i) {
                        int legal_id = legal_actions[i];
                        if (legal_id != -1) {
                            mask_ptr[legal_id] = true;
                        }
                    }

                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(obs_tensor);
                    inputs.push_back(mask_tensor);

                    torch::Tensor logits;
                    try {
                        auto output_tuple = ai_model.forward(inputs).toTuple();
                        logits = output_tuple->elements()[0].toTensor();
                    } catch (const std::exception& e) {
                        std::ofstream out("crash.txt");
                        out << "Exception in AI forward pass: " << e.what() << std::endl;
                        exit(1);
                    }

                    action_id = logits.argmax(1).item<int>();

                    // Wait for the AI to "think" so it doesn't play instantly right after the previous player
                    // (passing picks are quick: 9 of them happen back to back)
                    std::this_thread::sleep_for(std::chrono::milliseconds(is_passing ? 150 : 800));
                }

                if (!is_passing && env.GetState().current_trick.size() == 3) {
                    trick_override = env.GetState().current_trick;
                    Card c;
                    c.suit = static_cast<Suit>(action_id / 13);
                    c.rank = (action_id % 13) + 2;
                    trick_override.push_back({current_player, c});
                    show_trick_override = true;
                    screen.PostEvent(Event::Custom);
                    
                    std::this_thread::sleep_for(std::chrono::seconds(2)); // Pause for 2s on 4th card
                    show_trick_override = false;
                }

                auto result = env.Step(action_id);

                if (is_passing && !env.IsPassing()) {
                    // The distribution just resolved: anything now in the human's
                    // hand that wasn't in the kept snapshot arrived via the pass
                    std::lock_guard<std::mutex> lk(pass_mutex);
                    received_cards.clear();
                    for (const auto& c : env.GetState().hands[0]) {
                        int id = (static_cast<int>(c.suit) * 13) + (c.rank - 2);
                        if (std::find(human_kept.begin(), human_kept.end(), id) == human_kept.end()) {
                            received_cards.push_back(id);
                        }
                    }
                }

                if (result.done) {
                    screen.PostEvent(Event::Custom); // Draw the final trick and scores
                    std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 second pause
                    
                    bool hit_100 = false;
                    for (int i = 0; i < 4; ++i) {
                        if (env.GetState().total_scores[i] >= 100) hit_100 = true;
                    }
                    AppendMatchLog(data_prefix, env.GetState(), hit_100);
                    if (hit_100) {
                        overall_game_over = true;
                        game_over = true;
                        screen.PostEvent(Event::Custom);
                        continue;
                    } else {
                        {
                            std::lock_guard<std::mutex> lk(pass_mutex);
                            pass_selection.clear();
                            human_kept.clear();
                            received_cards.clear();
                        }
                        pass_confirmed = false;
                        env.Reset(); // Deal new round
                        screen.PostEvent(Event::Custom);
                        continue;
                    }
                }
                
                screen.PostEvent(Event::Custom);
            }
        }
        screen.PostEvent(Event::Custom);
    });

    auto event_handler = CatchEvent(renderer, [&](Event e) {
        if (e == Event::Character('g') || e == Event::Character('G')) {
            god_mode = !god_mode;
            return true;
        }
        return false;
    });

    screen.Loop(event_handler);
    
    if (game_thread.joinable()) {
        game_thread.join();
    }

    return 0;
}
