/* One avatar per CLINC150 domain, so the assistant that answers a banking
 * question is visibly not the one that answers a travel question.
 *
 * The groupings mirror `SELECTED` in train/prepare_data.py, which is where the
 * 40-intent subset is actually defined. Each bot-avatars type carries its own
 * palette colour, so the shapes are also colour-coded by domain.
 */

const DOMAIN_AVATARS = {
  small_talk: { type: 'clover', label: 'Small talk' },
  utility: { type: 'star', label: 'Utility' },
  travel: { type: 'triangle', label: 'Travel' },
  auto_commute: { type: 'drop', label: 'Auto & commute' },
  banking: { type: 'hexagon', label: 'Banking' },
  home: { type: 'flower', label: 'Home' },
  work: { type: 'square', label: 'Work' },
  kitchen_dining: { type: 'cat', label: 'Kitchen & dining' },
  out_of_scope: { type: 'ghost', label: 'Out of scope' },
}

const INTENT_DOMAIN = {
  // small talk
  greeting: 'small_talk',
  goodbye: 'small_talk',
  thank_you: 'small_talk',
  tell_joke: 'small_talk',
  what_is_your_name: 'small_talk',
  how_old_are_you: 'small_talk',
  are_you_a_bot: 'small_talk',
  what_can_i_ask_you: 'small_talk',
  // utility
  weather: 'utility',
  time: 'utility',
  date: 'utility',
  alarm: 'utility',
  timer: 'utility',
  definition: 'utility',
  calculator: 'utility',
  flip_coin: 'utility',
  // travel
  flight_status: 'travel',
  book_flight: 'travel',
  book_hotel: 'travel',
  translate: 'travel',
  exchange_rate: 'travel',
  // auto & commute
  directions: 'auto_commute',
  traffic: 'auto_commute',
  distance: 'auto_commute',
  gas: 'auto_commute',
  // banking
  balance: 'banking',
  transactions: 'banking',
  pay_bill: 'banking',
  credit_score: 'banking',
  // home
  play_music: 'home',
  next_song: 'home',
  shopping_list: 'home',
  todo_list: 'home',
  reminder: 'home',
  // work
  payday: 'work',
  meeting_schedule: 'work',
  pto_balance: 'work',
  // kitchen & dining
  recipe: 'kitchen_dining',
  restaurant_suggestion: 'kitchen_dining',
  calories: 'kitchen_dining',
  // the refusal class
  oos: 'out_of_scope',
}

export function avatarFor(intent) {
  const domain = INTENT_DOMAIN[intent] || 'out_of_scope'
  return { domain, ...DOMAIN_AVATARS[domain] }
}

export { DOMAIN_AVATARS, INTENT_DOMAIN }
