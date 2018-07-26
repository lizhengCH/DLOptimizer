from string import ascii_lowercase as letters
import tensorflow as tf


class Optimizer:
    def __init__(self, lr):
        self.alpha = lr

    @staticmethod
    def eye(x, y, padding=True):
        if padding:
            shape = tf.to_int32(x.get_shape())
            element = tf.eye(tf.reduce_prod(shape))
            return tf.reshape(element, tf.concat([shape, shape], 0))
        else:
            x_shape = tf.to_int32(x.get_shape())
            y_shape = tf.to_int32(y.get_shape())
            return tf.zeros(tf.concat([x_shape, y_shape], 0), dtype=tf.float32)

    @staticmethod
    def eye_like(x):
        shape = x.get_shape().as_list()
        half_shape = tf.constant(shape[:shape.__len__() // 2])

        element = tf.eye(tf.reduce_prod(half_shape))
        return tf.reshape(element, shape)

    @staticmethod
    def tensor_prod(x, y):
        lx = x.get_shape().as_list().__len__()
        ly = y.get_shape().as_list().__len__()
        wx = letters[:lx]
        wy = letters[lx:lx + ly]
        wxy = letters[:lx + ly]
        return tf.einsum('%s,%s->%s' % (wx, wy, wxy), x, y)

    @staticmethod
    def tensor_transuvection(x, y, mode='all'):
        x_shape = x.get_shape().as_list()
        y_shape = y.get_shape().as_list()
        lx = x_shape.__len__()
        ly = y_shape.__len__()
        if mode == 'all':
            assert x_shape == y_shape
            w = letters[:lx]
            return tf.einsum('%s,%s->' % (w, w), x, y)
        elif mode == 'right':
            assert x_shape[lx - ly:] == y_shape
            wx = letters[:lx]
            wy = letters[lx - ly:lx]
            wxy = letters[:lx - ly]
            return tf.einsum('%s,%s->%s' % (wx, wy, wxy), x, y)
        elif mode == 'auto':
            for i in range(1, min(lx, ly) + 1):
                if x_shape[lx - i:] == y_shape[:i]:
                    wx = letters[:lx]
                    wy = letters[lx - i:lx + ly - i]
                    wxy = letters[:lx - i] + letters[lx:lx + ly - i]
                    return tf.einsum('%s,%s->%s' % (wx, wy, wxy), x, y)
        else:
            raise ValueError('mode must in ["all", "right", "auto"]')

    def vec_outer_prod(self, xs, ys):
        return [[self.tensor_prod(x, y) for y in ys] for x in xs]

    def vec_inner_prod(self, xs, ys, mode):
        """
        set mode = 'all':
            if tensor in xs has the same shape as the tensor in ys
        set mode = 'right' or mode = 'auto:
            if tensor in xs has different shape of the tensor in ys
        """
        return tf.add_n([self.tensor_transuvection(x, y, mode=mode) for x, y in zip(xs, ys)])

    def mat_vec_prod(self, ms, xs, mode):
        return [self.vec_inner_prod(m, xs, mode=mode) for m in ms]

    def mat_mat_prod(self, mxs, mys, mode='auto'):
        ans = [[0 for _ in range(mys[0].__len__())] for _ in range(mxs.__len__())]
        for j in range(mys[0].__len__()):
            vec = self.mat_vec_prod(mxs, [mys[i][j] for i in range(mys.__len__())], mode=mode)
            for i in range(ans.__len__()):
                ans[i][j] = vec[i]
        return ans

    def build_step_length(self, f, xk, pk, shift):
        with tf.variable_scope('step_length', reuse=True):
            archive_xk = [tf.Variable(initial_value=xk[i], trainable=False, name='archive_xk_%d' % i)
                          for i in range(xk.__len__())]
            reset_xk = [x.assign(ax) for x, ax in zip(xk, archive_xk)]
            reset_archive_xk = [ax.assign(x) for ax, x in zip(archive_xk, xk)]

            archive_pk = [tf.Variable(initial_value=pk[i], trainable=False, name='archive_pk_%d' % i)
                          for i in range(pk.__len__())]
            reset_archive_pk = [ap.assign(p) for ap, p in zip(archive_pk, pk)]

            reset_x = tuple(reset_xk)
            reset_axp = tf.tuple(reset_archive_xk + reset_archive_pk)

            with tf.control_dependencies(
                    [x.assign(ax + tf.multiply(shift, ap)) for x, ax, ap in zip(xk, archive_xk, pk)]):
                f_shift = tf.identity(f)

            gradients = tf.gradients(f, xk)
            with tf.control_dependencies(
                    [x.assign(ax + tf.multiply(shift, ap)) for x, ax, ap in zip(xk, archive_xk, pk)]):
                gk = [tf.identity(g) for g in gradients]
                g_shift_pk = self.vec_inner_prod(gk, pk, mode='all')

            return reset_x, reset_axp, f_shift, g_shift_pk

    def step_length(self, sess, shift, feed_dict, reset_x, reset_axp, f_shift, g_shift_pk, alpha=1.0, is_newton=False):
        # reset archive_xk and archive_pk
        sess.run(reset_axp, feed_dict=feed_dict)

        low = 0.0
        high = 1.0

        c1 = 1e-4
        c2 = 0.9 if is_newton else 0.1

        f_0 = sess.run(f_shift, feed_dict={shift: 0., **feed_dict})
        g_pk = sess.run(g_shift_pk, feed_dict={shift: 0., **feed_dict})
        for i in range(10):
            f_alpha = sess.run(f_shift, feed_dict={shift: alpha, **feed_dict})
            if i == 0:
                g_alpha_pk = sess.run(g_shift_pk, feed_dict={shift: alpha, **feed_dict})

            cond1 = f_alpha <= f_0 + c1 * alpha * g_pk
            cond2 = abs(g_alpha_pk) <= c2 * abs(g_pk)
            if cond1 and cond2:
                sess.run(reset_x)
                return alpha

            f_high = sess.run(f_shift, feed_dict={shift: high, **feed_dict})
            f_low = sess.run(f_shift, feed_dict={shift: low, **feed_dict})
            g_low_pk = sess.run(g_shift_pk, feed_dict={shift: low, **feed_dict})

            alpha = - g_low_pk * (high ** 2) / 2 / (f_high - f_low - g_low_pk * high)
            if alpha < low or alpha > high:
                alpha = (low + high) / 2

            g_alpha_pk = sess.run(g_shift_pk, feed_dict={shift: alpha, **feed_dict})
            if g_alpha_pk > 0:
                high = alpha
            elif g_alpha_pk <= 0:
                low = alpha
            # print(i)

        sess.run(reset_x)
        return alpha


class CGOptimizer(Optimizer):
    def minimize(self, loss, var_list=None):
        if var_list is None:
            var_list = tf.trainable_variables()

        with tf.name_scope('conjugate_gradient'):
            grads = tf.gradients(loss, var_list)

            norm_gk1 = tf.add_n([tf.reduce_sum(tf.pow(g, 2)) for g in grads])
            norm_gk = tf.Variable(initial_value=tf.add_n([tf.reduce_sum(tf.pow(g, 2)) for g in grads]),
                                  trainable=False, name='norm_gk', dtype=tf.float32)
            pk = [tf.Variable(initial_value=-g, trainable=False) for g in grads]

            assign_xk = [x.assign_add(self.alpha * p) for x, p in zip(var_list, pk)]
            with tf.control_dependencies(assign_xk):
                assign_pk = [p.assign(tf.divide(norm_gk, norm_gk1) * p - g) for p, g in zip(pk, grads)]
                with tf.control_dependencies(assign_pk):
                    return norm_gk.assign(norm_gk1)


class ACGOptimizer(CGOptimizer):
    def __init__(self, loss, lr, var_list=None):
        super(CGOptimizer, self).__init__(lr)

        self.loss = loss
        self.xk = var_list if var_list is not None else tf.trainable_variables()
        self.pk = [tf.negative(g) for g in tf.gradients(self.loss, self.xk)]
        self.shift = tf.placeholder(dtype=tf.float32, shape=[], name='shift')

        self.reset_xp, self.reset_axp, self.f_shift, self.g_shift_pk = self.build_step_length(
            self.loss, self.xk, self.pk, self.shift)

    def auto_minimize(self):
        return self.minimize(self.loss, var_list=self.xk)

    def get_step_length(self, sess, feed_dict):
        reset_x, reset_axp, f_shift, g_shift_pk = self.build_step_length(self.loss, self.xk, self.pk, self.shift)
        return self.step_length(sess, self.shift, feed_dict, reset_x, reset_axp, f_shift, g_shift_pk, is_newton=False)


class BFGSOptimizer(Optimizer):
    def minimize(self, loss, var_list=None):
        if var_list is None:
            var_list = tf.trainable_variables()

        with tf.name_scope('bfgs'):
            gk1 = tf.gradients(loss, var_list)

            xk = var_list
            Hk = [[tf.Variable(initial_value=self.eye(y, x, x == y), trainable=False) for x in var_list]
                  for y in var_list]
            gk = [tf.Variable(initial_value=g, trainable=False) for g in gk1]

            #  this collection will be used in ABFGSOptimizer
            for hg in self.mat_vec_prod(Hk, gk, mode='right'):
                tf.add_to_collection('BFGS_pk', -hg)

            sk = [-self.alpha * hg for hg in self.mat_vec_prod(Hk, gk, mode='right')]
            assign_xk = [x.assign_add(s) for x, s in zip(xk, sk)]
            with tf.control_dependencies(assign_xk):
                yk = [g1 - g for g1, g in zip(gk1, gk)]
                rho_k = 1 / self.vec_inner_prod(yk, xk, mode='all')

                n = var_list.__len__()

                mat_sy = self.vec_outer_prod(sk, yk)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            mat_sy[i][j] = self.eye_like(mat_sy[i][j]) - rho_k * mat_sy[i][j]
                        else:
                            mat_sy[i][j] = -mat_sy[i][j]

                mat_ys = self.vec_outer_prod(yk, sk)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            mat_ys[i][j] = self.eye_like(mat_ys[i][j]) - rho_k * mat_ys[i][j]
                        else:
                            mat_ys[i][j] = -mat_ys[i][j]

                mat_ss = self.vec_outer_prod(sk, sk)
                for i in range(n):
                    for j in range(n):
                        mat_ss[i][j] = rho_k * mat_ss[i][j]

                Hk1 = self.mat_mat_prod(self.mat_mat_prod(mat_sy, Hk, mode='auto'), mat_ys, mode='auto')
                for i in range(n):
                    for j in range(n):
                        Hk1[i][j] += mat_ss[i][j]

                assign_Hk = [Hk[i][j].assign(Hk1[i][j]) for j in range(n) for i in range(n)]

                with tf.control_dependencies(assign_Hk):
                    return tf.tuple([g.assign(g1) for g, g1 in zip(gk, gk1)])


class ABFGSOptimizer(BFGSOptimizer):  # TODO: This class is not really work yet!
    def __init__(self, loss, lr, var_list=None):
        super(ABFGSOptimizer, self).__init__(lr)

        self.loss = loss
        self.xk = var_list if var_list is not None else tf.trainable_variables()
        self.pk = tf.get_collection('BFGS_pk')
        self.shift = tf.placeholder(dtype=tf.float32, shape=[], name='shift')

        self.reset_xp, self.reset_axp, self.f_shift, self.g_shift_pk = self.build_step_length(
            self.loss, self.xk, self.pk, self.shift)

    def auto_minimize(self):
        return self.minimize(self.loss, var_list=self.xk)

    def get_step_length(self, sess, feed_dict):
        reset_x, reset_axp, f_shift, g_shift_pk = self.build_step_length(self.loss, self.xk, self.pk, self.shift)
        return self.step_length(sess, self.shift, feed_dict, reset_x, reset_axp, f_shift, g_shift_pk, is_newton=True)
