# -*- coding: utf-8 -*-
"""Split preprocessing mixins."""

from scripts.preprocess import base as _base

# Reuse shared namespace (imports/constants/helpers) from base module.
globals().update({k: v for k, v in _base.__dict__.items() if k not in globals()})


class EvalGridMixin:
    """Mixin with grouped preprocessing methods."""

    def gen_eval_grid(self, ocean_mask: bool = True, region_mask: bool = True) -> str:
        """Generate the evaluation grid based on recipe specifications.

        Args:
            ocean_mask: Whether to include an ocean/land mask.
            region_mask: Whether to include the NSIDC region mask.

        Returns:
            Path to the generated grid NetCDF file.
        """
        env_key = f'SITOOL_EVAL_GRID_{str(self.hemisphere).upper()}'
        env_grid = str(os.environ.get(env_key, '')).strip()
        if env_grid and os.path.exists(env_grid):
            return env_grid

        hms_str = 'N' if self.hemisphere == 'nh' else 'S'
        aux_dir = os.path.join(self.ref_data_path, 'Auxiliary')

        etopo_file = None
        if ocean_mask:
            p = os.path.join(aux_dir, 'ETOPO_2022_v1_60s_N90W180_surface.nc')
            if os.path.exists(p):
                etopo_file = p

        region_file = None
        if region_mask:
            p = os.path.join(aux_dir,
                 f'NSIDC-0780_SeaIceRegions_PS-{hms_str}3.125km_v1.0.nc')
            if os.path.exists(p):
                region_file = str(p)

        # Multiple modules may request the same evaluation grid concurrently.
        # Guard both .nc and .txt companion files with a shared path lock.
        proj_tag = str(self.grid_info['proj']).replace(':', '')
        grid_base = f"{proj_tag}_{self.hemisphere}_{self.grid_info['res']}"
        expected_nc = os.path.join(self.eval_dir, f'{grid_base}.nc')
        expected_txt = os.path.join(self.eval_dir, f'{grid_base}.txt')
        with _acquire_path_locks([expected_nc, expected_txt]):
            grid_file = utils.gen_polar_grid(
                hms=self.hemisphere,
                res=self.grid_info['res'],
                proj_name=self.grid_info['proj'],
                grid_path=self.eval_dir + '/',
                etopo_file=etopo_file,
                region_mask_file=region_file,
                ellps='WGS84',
            )
        return grid_file

    def _prepare_common(self, frequency: str, output_dir: str):
        """Common setup steps shared by prep_obs and prep_models.

        Includes: frequency validation, evaluation grid generation,
        date-range construction, and CDO instance creation.

        Returns:
            (grid_txt, start_date, end_date, cdo, module_vars)
        """
        if frequency not in ('daily', 'monthly'):
            raise ValueError("frequency must be 'daily' or 'monthly'")

        module_vars = self.recipe_vars[self.module_name]

        # Generate the evaluation grid (.nc) and its CDO description file (.txt)
        grid_file = self.gen_eval_grid()
        grid_txt = grid_file.replace('.nc', '.txt')

        os.makedirs(output_dir, exist_ok=True)

        # Build start/end date strings from the year_range in the recipe
        year_range = module_vars['year_range']
        start_date = f"{year_range[0]}-01-01"
        end_date = f"{year_range[1]}-12-31"

        # Create a fresh CDO instance per call to avoid stale state from a previous module
        cdo = self._new_cdo_instance()

        return grid_txt, start_date, end_date, cdo, module_vars

__all__ = [
    "EvalGridMixin",
]
