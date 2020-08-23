import torch

def svd_orthogonalization(lyr):
    r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        weights = lyr.weight.data.clone()
        c_out, c_in, f1, f2 = weights.size()
        dtype = lyr.weight.data.type()

        # Reshape filters to columns
        # From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
        weights = weights.permute(2,3, 1, 0).contiguous().view(f1 * f2 * c_in, c_out)

        try:
            # SVD decomposition and orthogonalization
            mat_u, _, mat_v = torch.svd(weights)
            weights = torch.mm(mat_u, mat_v.t())

            lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).type(dtype)
        except:
            pass
    else:
        pass
