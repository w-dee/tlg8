#include "tlg8_predictors.h"

namespace tlg::v8::enc
{
  Cas8Predictor::Cas8Predictor() : core_(CAS8::Config{}, 0, 255)
  {
  }

  void Cas8Predictor::reset(uint32_t components)
  {
    components_ = components;
    for (auto &state : states_)
      state = CAS8::State{};
  }

  std::pair<uint8_t, CAS8::PredId> Cas8Predictor::predict(uint32_t component,
                                                          uint8_t a,
                                                          uint8_t b,
                                                          uint8_t c,
                                                          uint8_t d,
                                                          uint8_t f)
  {
    if (component >= components_)
      return {0, CAS8::PredId::P2};
    const auto result = core_.predict8_and_choose(a, b, c, d, f, states_[component]);
    return {static_cast<uint8_t>(result.first), result.second};
  }

  void Cas8Predictor::update(uint32_t component, CAS8::PredId pid, int abs_error)
  {
    if (component >= components_)
      return;
    if (abs_error < 0)
      abs_error = -abs_error;
    core_.update_state(states_[component], pid, abs_error);
  }
}
