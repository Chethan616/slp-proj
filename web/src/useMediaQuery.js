import { useEffect, useState } from 'react'

/** Subscribe to a media query, so layout choices that are not purely CSS
 *  (which VoiceBeam geometry preset to use, say) can follow the viewport.
 *
 *  `query` is expected to be a constant string. The initial value is read
 *  during render and the listener carries every change after that, so the
 *  effect only subscribes - it never needs to set state on mount. */
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches,
  )

  useEffect(() => {
    const mq = window.matchMedia(query)
    const onChange = (e) => setMatches(e.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])

  return matches
}
