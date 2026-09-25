/* One avatar per part of the food domain, so the answer about a recipe is
 * visibly not the answer about a restaurant booking.
 *
 * The groupings mirror `SELECTED` in train/prepare_data.py, which is where the
 * 15-intent subset is defined. Each bot-avatars type carries its own palette
 * colour, so the shapes are colour-coded by area too.
 */

const AREA_AVATARS = {
  cooking: { type: 'flower', label: 'Cooking' },
  nutrition: { type: 'hexagon', label: 'Nutrition' },
  restaurants: { type: 'star', label: 'Restaurants' },
  out_of_scope: { type: 'ghost', label: 'Out of scope' },
}

const INTENT_AREA = {
  // cooking
  recipe: 'cooking',
  ingredients_list: 'cooking',
  ingredient_substitution: 'cooking',
  cook_time: 'cooking',
  meal_suggestion: 'cooking',
  // nutrition
  calories: 'nutrition',
  nutrition_info: 'nutrition',
  food_last: 'nutrition',
  // restaurants
  restaurant_suggestion: 'restaurants',
  restaurant_reviews: 'restaurants',
  how_busy: 'restaurants',
  restaurant_reservation: 'restaurants',
  confirm_reservation: 'restaurants',
  cancel_reservation: 'restaurants',
  accept_reservations: 'restaurants',
  // the refusal class
  oos: 'out_of_scope',
}

export function avatarFor(intent) {
  const area = INTENT_AREA[intent] || 'out_of_scope'
  return { area, ...AREA_AVATARS[area] }
}

export { AREA_AVATARS, INTENT_AREA }
