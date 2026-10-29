/**
 * Sherman's logo, the company's own artwork (brand/www.shermanep.com_*),
 * used as-is: the white version on dark backgrounds, the colour version on
 * light ones. Never redrawn or recoloured. `markOnly` shows just the "i"
 * mark (a crop of the same image) for narrow rails.
 */
import logoColour from '../assets/brand/sherman-logo-colour.png'
import logoWhite from '../assets/brand/sherman-logo-white.png'

/** Natural sizes of the two PNGs, and the width of the "i" mark in the white one. */
const WHITE = { w: 277, h: 57, mark: 41 }
const COLOUR = { w: 221, h: 46, mark: 34 }

export interface BrandLogoProps {
  /** Rendered height in CSS px (default: the image's natural height). */
  height?: number
  variant?: 'white' | 'colour'
  markOnly?: boolean
  className?: string
  alt?: string
}

export default function BrandLogo({ height, variant = 'white', markOnly = false, className, alt = 'Sherman' }: BrandLogoProps) {
  const art = variant === 'white' ? WHITE : COLOUR
  const h = height ?? art.h
  const w = Math.round((art.w * h) / art.h)
  const img = (
    <img
      src={variant === 'white' ? logoWhite : logoColour}
      width={w}
      height={h}
      alt={markOnly ? '' : alt}
      draggable={false}
      style={{ display: 'block', width: w, height: h, maxWidth: 'none' }}
    />
  )
  if (!markOnly) return <span className={className} style={{ display: 'inline-block', lineHeight: 0 }} data-brand-logo>{img}</span>
  return (
    <span
      className={className}
      role="img"
      aria-label={alt}
      data-brand-logo="mark"
      style={{ display: 'inline-block', lineHeight: 0, overflow: 'hidden', width: Math.round((art.mark * h) / art.h), height: h }}
    >
      {img}
    </span>
  )
}
