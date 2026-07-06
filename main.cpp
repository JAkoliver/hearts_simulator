#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <torch/script.h>
#include <torch/torch.h>
#include "HeartsEnv.hpp"

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

class CardButton : public ComponentBase {
public:
    std::string label;
    std::function<void()> on_click;
    bool is_legal;
    Suit suit;
    bool hovered = false;
    Box box_; // For mouse collision detection

    CardButton(std::string label_, std::function<void()> on_click_, bool is_legal_, Suit suit_)
        : label(std::move(label_)), on_click(std::move(on_click_)), is_legal(is_legal_), suit(suit_) {}

    Element Render() override {
        auto t = text(" " + label + " ") | border;
        
        bool is_focused = Focused();
        if (hovered || is_focused) t = t | inverted;
        
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

Component MakeCardButton(std::string label, std::function<void()> on_click, bool is_legal, Suit suit) {
    return Make<CardButton>(std::move(label), std::move(on_click), is_legal, suit);
}

int main() {
    // 1. Load the Model
    torch::jit::script::Module ai_model;
    try {
        ai_model = torch::jit::load("hearts_ai_grandmaster.pt");
        ai_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }

    // 2. Initialize Environment
    std::random_device rd;
    HeartsEnv env(rd());
    env.Reset();
    
    auto screen = ScreenInteractive::Fullscreen();
    
    std::atomic<bool> game_over(false);
    std::atomic<bool> overall_game_over(false);
    std::atomic<bool> show_trick_override(false);
    std::atomic<int> action_to_play(-1); // Thread synchronization variable
    std::vector<PlayedCard> trick_override;

    // 3. Human UI Component (Interactive Container)
    auto hand_container = Container::Horizontal({});

    auto renderer = Renderer(hand_container, [&] {
        const auto& state = env.GetState();
        
        if (overall_game_over) {
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
                text("Press Ctrl+C to exit.") | center | dim
            }) | center;
        }

        auto score_box = window(text(" Scores "), vbox({
            text("Player 0 (You): Total " + std::to_string(state.total_scores[0]) + " (Round: " + std::to_string(state.round_scores[0]) + ")"),
            text("Player 1 (AI):  Total " + std::to_string(state.total_scores[1]) + " (Round: " + std::to_string(state.round_scores[1]) + ")"),
            text("Player 2 (AI):  Total " + std::to_string(state.total_scores[2]) + " (Round: " + std::to_string(state.round_scores[2]) + ")"),
            text("Player 3 (AI):  Total " + std::to_string(state.total_scores[3]) + " (Round: " + std::to_string(state.round_scores[3]) + ")"),
        }));
        
        Elements trick_elements;
        const auto& active_trick = show_trick_override ? trick_override : state.current_trick;
        for (const auto& card : active_trick) {
            int card_id = static_cast<int>(card.card.suit) * 13 + (card.card.rank - 2);
            trick_elements.push_back(text("P" + std::to_string(card.player_id) + ": " + CardToString(card_id)));
        }
        if (trick_elements.empty()) {
            trick_elements.push_back(text("Waiting for lead..."));
        }
        
        auto trick_box = window(text(" Current Trick "), vbox(std::move(trick_elements)) | center);
        
        int current_player = env.GetCurrentPlayer();
        
        Element status_text;
        if (game_over) {
            status_text = text("Game Over! Press Ctrl+C to exit.") | center | bold;
        } else if (current_player == 0) {
            status_text = text("Your Turn (Select a Card to Play)") | center | bold;
        } else {
            status_text = text("AI is thinking... Player " + std::to_string(current_player) + "'s turn.") | center | bold;
        }
        
        auto bottom_panel = window(text(" Your Hand "), hand_container->Render() | center) | size(HEIGHT, EQUAL, 5);
        
        return vbox({
            score_box,
            filler(),
            trick_box,
            filler(),
            status_text,
            bottom_panel
        });
    });

    // 4. Game Logic Thread
    std::thread game_thread([&]() {
        while (!game_over) {
            int current_player = env.GetCurrentPlayer();
            auto legal_actions = env.GetLegalActions();
            
            std::vector<Card> current_hand = env.GetState().hands[0];
            
            screen.Post([&, current_player, legal_actions, current_hand]() {
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
                    
                    auto btn = MakeCardButton(CardToString(action_id), [&, action_id]() {
                        if (action_to_play == -1) action_to_play = action_id;
                    }, is_legal, c.suit);
                    
                    hand_container->Add(btn);
                }
            });
            
            screen.PostEvent(Event::Custom); // Force redraw
            
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
                    if (hit_100) {
                        overall_game_over = true;
                        game_over = true;
                        screen.PostEvent(Event::Custom);
                        break;
                    } else {
                        env.Reset(); // Deal new round
                        screen.PostEvent(Event::Custom);
                        continue;
                    }
                }
                screen.PostEvent(Event::Custom);
                
            } else {
                // AI Turn
                auto obs = env.Observe();
                
                torch::Tensor obs_tensor = torch::from_blob(obs.data(), {1, 181}, torch::kFloat32).clone();
                
                torch::Tensor mask_tensor = torch::zeros({1, 52}, torch::kBool);
                bool* mask_ptr = mask_tensor.data_ptr<bool>();
                for(int i = 0; i < 13; ++i) {
                    int action_id = legal_actions[i];
                    if (action_id != -1) {
                        mask_ptr[action_id] = true;
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
                
                int action_id = logits.argmax(1).item<int>();
                
                if (env.GetState().current_trick.size() == 3) {
                    trick_override = env.GetState().current_trick;
                    Card c;
                    c.suit = static_cast<Suit>(action_id / 13);
                    c.rank = (action_id % 13) + 2;
                    trick_override.push_back({current_player, c});
                    show_trick_override = true;
                    screen.PostEvent(Event::Custom);
                    
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    show_trick_override = false;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(800));
                }
                
                auto result = env.Step(action_id);
                if (result.done) {
                    screen.PostEvent(Event::Custom); // Draw the final trick and scores
                    std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 second pause
                    
                    bool hit_100 = false;
                    for (int i = 0; i < 4; ++i) {
                        if (env.GetState().total_scores[i] >= 100) hit_100 = true;
                    }
                    if (hit_100) {
                        overall_game_over = true;
                        game_over = true;
                        screen.PostEvent(Event::Custom);
                        break;
                    } else {
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

    screen.Loop(renderer);
    
    if (game_thread.joinable()) {
        game_thread.join();
    }

    return 0;
}
