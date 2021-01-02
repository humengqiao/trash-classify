import * as tf from '@tensorflow/tfjs'

export const file2img = file => {
  return new Promise(resolve => {
    const reader = new FileReader()
    reader.onload = event => {
      const img = document.createElement('img')
      img.src = event.target.result
      img.width = 224
      img.height = 224
      img.onload = () => resolve(img)
    }
    reader.readAsDataURL(file)
  })
}

export const img2Tensor = img => {
  return tf.tidy(() => {
    return tf.browser.fromPixels(img)
      .toFloat()
      .sub(255 / 2)
      .div(255 / 2)
      .reshape([1, 224, 224, 3])
  })
}