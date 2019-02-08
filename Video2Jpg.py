import av
container = av.open('video/IMG_1830.mp4')
for frame in container.decode(video=0):
    frame.to_image().save('video/frames/frame-%04d.bmp'% frame.index)


'''
# Signal that we only want to look at keyframes.
stream = container.streams.video[0]
stream.codec_context.skip_frame = 'NONKEY'

for frame in container.decode(stream):
    print(frame)
    # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
    frame.to_image().save(
        'video/frames/keyframe-{:04d}.jpg'.format(frame.pts),
        quality=80,
    )
'''