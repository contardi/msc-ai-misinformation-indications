export const PUBLIC_BUILD = import.meta.env.VITE_PUBLIC_BUILD === '1'

export const ACCEPTED_MEDIA = [
  'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a', 'audio/aac', 'audio/ogg',
  'audio/opus', 'audio/wav', 'audio/x-wav', 'audio/webm', 'audio/flac', 'audio/amr',
  'video/mp4', 'video/mpeg', 'video/quicktime', 'video/webm', 'video/x-matroska', 'video/3gpp',
].join(',')

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024
